import os
import json
import torch
import numpy as np
import argparse
import random
import networkx as nx
from tqdm import tqdm
import asyncio
import math
import copy
import sys
import time
import shutil

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from experiment.args import Args
from experiment.model import ARGDesigner
from experiment.train_ARGDesigner import train
from experiment.utils import load_model, generate_graph, convert_to_pyg_graph, Accuracy, get_kwargs, save_graph_with_features
from sentence_transformers import SentenceTransformer
from experiment import process_dataset as gdata

from mas_framework.graph.graph import Graph, TestGraph
from datasets.mmlu_dataset import MMLUDataset
from experiment.mmlu.mmlu_prompt_set import ROLE_DESCRIPTION

FINETUNE_DATA_DIR = "/root/ARG-Designer/experiment/FinetuneData_mmlu"
COLD_START_DIR = "/root/ARG-Designer/experiment/ColdStartData_mmlu"
TASK_SPLIT_FILE = "./task_split_mmlu.json"


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Full pipeline: pretrain, data generation, finetune")
    parser.add_argument('--pretrain', action='store_true', help="Run pretraining from scratch")
    parser.add_argument('--load_from_dir', type=str, default=None, help="Directory to load pretrained model")

    parser.add_argument('--pretrain_epochs', type=int, default=100, help="Number of pretraining epochs")
    parser.add_argument('--pretrain_lr', type=float, default=1e-4, help="Learning rate for pretraining")

    parser.add_argument('--pruning_ratio', type=float, default=0.25, help="Pruning ratio for data generation")
    parser.add_argument('--limit_generation_samples', type=int, default=999, help="Max samples for pruned data")
    parser.add_argument('--replay_ratio', type=float, default=0.3, help="Replay ratio from cold-start data")

    parser.add_argument('--finetune_epochs', type=int, default=100, help="Number of finetuning epochs")
    parser.add_argument('--finetune_lr', type=float, default=5e-5, help="Learning rate for finetuning")

    parser.add_argument('--llm_name', type=str, default="gpt-4o-mini", help="LLM model name")
    parser.add_argument('--domain', type=str, default="mmlu", help="Task domain")
    parser.add_argument('--decision_method', type=str, default="FinalRefer", help="Decision method for final node")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['AnalyzeAgent'], help="Names of agents")
    parser.add_argument('--num_rounds', type=int, default=1, help="Number of inference rounds")

    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training and generation")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--model_output_dir', type=str, default='output/efficiency_finetuned_model',
                        help="Output directory for all model artifacts")
    return parser.parse_args()


def setup_environment(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def get_unique_simple_configs():
    configs = set()
    for agent_num in range(2, 7):
        if agent_num == 2:
            configs.add(('Chain', 2))
        elif agent_num == 3:
            configs.add(('Chain', 3))
            configs.add(('Star', 3))
        else:
            configs.add(('Chain', agent_num))
            configs.add(('Star', agent_num))
            configs.add(('Layered', agent_num))
    return list(configs)


def run_pretraining(args):
    print("\n==================== Stage 1: Pretraining ====================")
    pretrain_args = Args().update_args()
    pretrain_args.data_dir = COLD_START_DIR
    pretrain_args.experiment_path = args.model_output_dir
    pretrain_args.pretrain = args.pretrain
    pretrain_args.epochs = args.pretrain_epochs
    pretrain_args.lr = args.pretrain_lr
    pretrain_args.batch_size = args.batch_size
    pretrain_args.seed = args.seed
    pretrain_args.model_name = 'best_model.pth'

    graph_dataset, _ = gdata.load_graph_dataset(pretrain_args)
    role_to_id = graph_dataset.role_to_id
    id_to_role = graph_dataset.id_to_role
    num_node_types = len(role_to_id) + 2
    pretrain_args.role_mapping = role_to_id
    pretrain_args.id_to_role = id_to_role
    pretrain_args.START_TOKEN_ID = len(role_to_id)
    pretrain_args.END_TOKEN_ID = len(role_to_id) + 1

    data_statistics = gdata.get_data_statistics(graph_dataset)
    data_statistics['num_node_labels'] = num_node_types
    data_statistics['num_edge_labels'] = 1

    correct_graphs = [g for g in graph_dataset if g.graph.get('is_correct')]
    incorrect_graphs = [g for g in graph_dataset if not g.graph.get('is_correct')]
    random.shuffle(correct_graphs)
    random.shuffle(incorrect_graphs)

    train_ratio = 0.9
    train_graphs = correct_graphs[:int(len(correct_graphs) * train_ratio)] + incorrect_graphs[:int(len(incorrect_graphs) * train_ratio)]
    val_graphs = correct_graphs[int(len(correct_graphs) * train_ratio):] + incorrect_graphs[int(len(incorrect_graphs) * train_ratio):]
    random.shuffle(train_graphs)
    random.shuffle(val_graphs)

    print(f"Train set: {len(train_graphs)}, Validation set: {len(val_graphs)}")
    dataset_train = gdata.GraphListDataset(train_graphs, pretrain_args)
    dataset_validate = gdata.GraphListDataset(val_graphs, pretrain_args)

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=pretrain_args.batch_size, shuffle=True, drop_last=True,
        num_workers=pretrain_args.num_workers, collate_fn=lambda _: _
    )
    dataloader_validate = torch.utils.data.DataLoader(
        dataset_validate, batch_size=pretrain_args.batch_size, shuffle=False, drop_last=False,
        num_workers=pretrain_args.num_workers, collate_fn=lambda _: _
    )

    with open(os.path.join(pretrain_args.experiment_path, "configuration.txt"), 'w') as f:
        json.dump(pretrain_args.__dict__, f, indent=2)

    model = ARGDesigner(pretrain_args, data_statistics).to(pretrain_args.device)
    train(pretrain_args, model, dataloader_train, dataloader_validate)

    print("==================== Stage 1 complete ====================")
    return os.path.join(pretrain_args.experiment_path), os.path.join(pretrain_args.experiment_path, 'configuration.txt')


def apply_efficiency_strategy(graphs, strategy='prune', **kwargs):
    graph_list = []
    for graph in graphs:
        if strategy == 'prune':
            pruning_ratio = kwargs.get('pruning_ratio', 0.2)
            num_edges_to_remove = int(graph.number_of_edges() * pruning_ratio)
            if num_edges_to_remove > 0:
                edges = list(graph.edges())
                random.shuffle(edges)
                edges_to_remove = edges[:num_edges_to_remove]
                graph.remove_edges_from(edges_to_remove)
                if not nx.is_weakly_connected(graph):
                    components = list(nx.weakly_connected_components(graph))
                    if components:
                        main_component = max(components, key=len)
                        for comp in components:
                            if comp != main_component:
                                graph.add_edge(random.choice(list(comp)), random.choice(list(main_component)))
        graph_list.append(graph)
    return graph_list


async def generate_pruned_data(args, dataset, model_path, output_dir):
    print("\n==================== Stage 2a: Generate Pruned Data ====================")
    model = load_model(model_path)
    model.eval()
    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    accuracy = Accuracy()
    role_constraints_dict = ROLE_DESCRIPTION
    original_dataset = dataset.dataset

    successful = 0
    total = min(args.limit_generation_samples, len(dataset))
    records = [(i, record) for i, record in enumerate(dataset) if i < total]

    with tqdm(total=total, desc="Pruning data") as pbar:
        for i in range(0, len(records), args.batch_size):
            batch = records[i:i + args.batch_size]
            tasks = []
            for record_idx, record in batch:
                input_dict = original_dataset.record_to_input(record)
                task_text = input_dict['task']
                task_emb = torch.tensor(sentence_model.encode(task_text),
                                         device=model.args.device).float()
                gen_graph = generate_graph(model, task_emb, role_constraints_dict, record_idx)
                eff_graphs = apply_efficiency_strategy(gen_graph, pruning_ratio=args.pruning_ratio)
                if eff_graphs:
                    g = eff_graphs[0]
                    pyg = convert_to_pyg_graph(g, task_text)
                    tg = TestGraph(domain=args.domain, llm_name=args.llm_name,
                                   decision_method=args.decision_method, pyg_data=pyg)
                    tasks.append((tg.arun(input_dict, num_rounds=1), {"pyg": pyg, "record":record, "idx": record_idx, "text": task_text}))
            if not tasks:
                continue
            coros = [t[0] for t in tasks]
            results = await asyncio.gather(*coros, return_exceptions=True)
            for result, (_, meta) in zip(results, tasks):
                pbar.update(1)
                if isinstance(result, Exception):
                    continue
                answer = original_dataset.postprocess_answer(result)
                correct = original_dataset.record_to_target_answer(meta["record"])
                is_corr = accuracy.update(answer, correct)
                if is_corr:
                    successful += 1
                    rid = meta["idx"]
                    fname = f"eff_q{rid}_g0.pt"
                    fpath = os.path.join(output_dir, fname)
                    setattr(meta["pyg"], 'is_correct', True)
                    setattr(meta["pyg"], 'question', meta["text"])
                    setattr(meta["pyg"], 'mode', 'EfficientPruned')
                    torch.save(meta["pyg"], fpath)

    print(f"Stage 2a complete: {successful} pruned graphs saved.")


async def generate_simple_data(args, dataset, output_dir):
    print("\n==================== Stage 2b: Generate Simple Data ====================")
    configs = get_unique_simple_configs()
    for mode, num in configs:
        kwargs = get_kwargs(mode, num)
        roles = random.choices(list(ROLE_DESCRIPTION.keys()), k=num)
        kwargs['node_kwargs'] = [{'role': r} for r in roles]
        graph = Graph(domain=args.domain, llm_name=args.llm_name,
                      agent_names=[args.agent_names[0]] * num,
                      decision_method=args.decision_method, **kwargs)
        await evaluate_and_save(graph, dataset, args, mode, num, output_dir)
    print("Stage 2b complete.")


def generate_replay_data(args, output_dir):
    print("\n==================== Stage 2c: Generate Replay Data ====================")
    cold_path = COLD_START_DIR
    if not os.path.isdir(cold_path):
        print(f"Error: Cold-start data directory '{cold_path}' not found.")
        return
    files = [f for f in os.listdir(cold_path) if f.endswith("True.pt") and os.path.isfile(os.path.join(cold_path, f))]
    if not files:
        print(f"Warning: No successful graphs found for replay in '{cold_path}'.")
        return
    sample_n = min(int(len(files) * args.replay_ratio), len(files))
    to_copy = random.sample(files, sample_n)
    for fn in tqdm(to_copy, desc="Copying replay data"):
        shutil.copy(os.path.join(cold_path, fn), os.path.join(output_dir, fn))
    print("Stage 2c complete.")


async def evaluate_and_save(graph: Graph, dataset, args, current_mode: str, current_agent_num: int, output_dir: str):
    accuracy = Accuracy()
    original_dataset = dataset.dataset
    records = list(enumerate(dataset))
    num_batches = math.ceil(len(records) / args.batch_size)
    for i in tqdm(range(num_batches), desc=f"Eval {current_mode}-{current_agent_num}"):
        batch = records[i * args.batch_size:(i + 1) * args.batch_size]
        tasks = []
        for idx, record in batch:
            g_copy = copy.deepcopy(graph)
            inp = original_dataset.record_to_input(record)
            flow = g_copy.to_pyg_graph(inp)
            tg = TestGraph(domain=args.domain, llm_name=args.llm_name,
                           decision_method=args.decision_method, pyg_data=flow)
            tasks.append((tg.arun(inp, args.num_rounds), {"flow": flow, "idx": idx, "question": inp['task']}))
        coros = [t[0] for t in tasks]
        results = await asyncio.gather(*coros, return_exceptions=True)
        for res, (_, meta) in zip(results, tasks):
            if isinstance(res, Exception):
                continue
            ans = original_dataset.postprocess_answer(res)
            corr = original_dataset.record_to_target_answer(dataset[meta["idx"]])
            is_corr = accuracy.update(ans, corr)
            if is_corr:
                rid = meta["idx"]
                name = f"mmlu_{rid}_{current_mode}_{current_agent_num}_{is_corr}"
                path = os.path.join(output_dir, f"{name}.pt")
                save_graph_with_features(meta["flow"], path, {
                    "mode": current_mode,
                    "agent_nums": current_agent_num,
                    "is_correct": is_corr,
                    "question": meta["question"]
                })
    accuracy.print()
    print(f"Finished {current_mode}-{current_agent_num}, accuracy: {accuracy.get():.2f}%")
    return accuracy.get()


def run_finetuning(model_path, config_path, finetune_data_dir, args):
    print("\n==================== Stage 3: Finetuning ====================")
    finetune_args = Args().update_args()
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    for k, v in cfg.items():
        if k != 'dataset':
            setattr(finetune_args, k, v)
    finetune_args.data_dir_ef = finetune_data_dir
    finetune_args.epochs = args.finetune_epochs
    finetune_args.lr = args.finetune_lr
    finetune_args.batch_size = args.batch_size
    finetune_args.data_dir = COLD_START_DIR
    finetune_args.seed = args.seed
    finetune_args.experiment_path = args.model_output_dir
    finetune_args.pretrain = False
    finetune_args.model_name = 'ef_best_model.pth'

    efficient_dataset, _ = gdata.load_graph_dataset(finetune_args, pretrain=False)
    if not efficient_dataset.graph_list:
        print("No finetune data found, skipping finetune.")
        return

    stats = gdata.get_data_statistics(efficient_dataset)
    stats['num_node_labels'] = len(efficient_dataset.role_to_id) + 2
    stats['num_edge_labels'] = 1

    ds = gdata.GraphListDataset(efficient_dataset, finetune_args)
    dl = torch.utils.data.DataLoader(ds, batch_size=finetune_args.batch_size, shuffle=True, collate_fn=lambda x: x)
    model = ARGDesigner(finetune_args, stats).to(finetune_args.device)
    ckpt = torch.load(os.path.join(model_path, 'best_model.pth'))
    model.load_state_dict(ckpt['model_state_dict'])
    print("Loaded pretrained weights, starting finetune...")
    train(finetune_args, model, dl)
    print("==================== Stage 3 complete ====================")


def find_latest_model_dir(base_dir='output'):
    prefix = 'efficiency_finetuned_model'
    dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith(prefix)]
    return os.path.join(base_dir, sorted(dirs)[-1]) if dirs else None


async def main(flag):
    args = parse_cli_args()
    args.pretrain = flag
    setup_environment(args)
    loop = asyncio.get_running_loop()

    if args.pretrain:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        args.model_output_dir = f"{args.model_output_dir}_{timestamp}"
        os.makedirs(args.model_output_dir, exist_ok=True)
        print(f"Starting new run, outputs in: {args.model_output_dir}")
        pretrained_path, config_path = await loop.run_in_executor(None, run_pretraining, args)
    else:
        model_dir = args.load_from_dir or find_latest_model_dir()
        if not model_dir:
            raise FileNotFoundError("Model directory not found. Run with --pretrain or specify --load_from_dir.")
        args.model_output_dir = model_dir
        pretrained_path = model_dir
        config_path = os.path.join(model_dir, "configuration.txt")
        print(f"Loading existing model from: {model_dir}")

    os.makedirs(FINETUNE_DATA_DIR, exist_ok=True)
    if not os.path.exists(TASK_SPLIT_FILE):
        raise FileNotFoundError(f"Task split file '{TASK_SPLIT_FILE}' not found.")
    with open(TASK_SPLIT_FILE, 'r') as f:
        ts = json.load(f)
    finetune_indices = ts['finetune_tasks']
    dataset = MMLUDataset('dev')
    finetune_subset = torch.utils.data.Subset(dataset, finetune_indices)
    print(f"Loaded {len(finetune_subset)} tasks for finetune data generation.")

    await generate_pruned_data(args, finetune_subset, pretrained_path, FINETUNE_DATA_DIR)
    await generate_simple_data(args, finetune_subset, FINETUNE_DATA_DIR)
    generate_replay_data(args, FINETUNE_DATA_DIR)

    await loop.run_in_executor(None, run_finetuning, pretrained_path, config_path, FINETUNE_DATA_DIR, args)
    print("\nPipeline complete. Final model at:", args.model_output_dir)


if __name__ == '__main__':
    asyncio.run(main(True))
