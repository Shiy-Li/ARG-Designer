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
import shutil
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from experiment.args import Args
from experiment.model import ARGDesigner
from experiment.train_ARGDesigner import train
from experiment.utils import load_model, generate_graph, convert_to_pyg_graph, get_kwargs, save_graph_with_features
from sentence_transformers import SentenceTransformer
from experiment import process_dataset as gdata
from mas_framework.graph.graph import Graph, TestGraph
from mas_framework.tools.reader.readers import JSONLReader
from mas_framework.tools.coding.python_executor import PyExecutor
from experiment.humaneval.humaneval_prompt_set import ROLE_DESCRIPTION

FINETUNE_DATA_DIR = "/root/ARG-Designer/experiment/FinetuneData_humaneval"
COLD_START_DIR = "/root/ARG-Designer/experiment/ColdStartData_humaneval"
TASK_SPLIT_FILE = "./task_split_humaneval.json"

def parse_cli_args():
    parser = argparse.ArgumentParser(description="pretrain->data generation->finetune")
    parser.add_argument('--pretrain', action='store_true', help="If set, run pretraining from scratch; otherwise load existing model.")
    parser.add_argument('--load_from_dir', type=str, default=None, help="Directory to load pretrained model from; if None, find latest automatically.")

    parser.add_argument('--pretrain_epochs', type=int, default=100, help="Number of epochs for pretraining.")
    parser.add_argument('--pretrain_lr', type=float, default=8e-5, help="Learning rate for pretraining.")

    parser.add_argument('--pruning_ratio', type=float, default=0.25, help="Edge pruning ratio for efficiency strategy.")
    parser.add_argument('--limit_generation_samples', type=int, default=9999, help="Max samples for pruned data generation.")
    parser.add_argument('--replay_ratio', type=float, default=0.3, help="Sampling ratio from cold-start data for replay.")

    parser.add_argument('--finetune_epochs', type=int, default=100, help="Number of epochs for finetuning.")
    parser.add_argument('--finetune_lr', type=float, default=1e-5, help="Learning rate for finetuning.")

    parser.add_argument('--llm_name', type=str, default="gpt-4o-mini", help="Name of the LLM to use.")
    parser.add_argument('--domain', type=str, default="humaneval", help="Task domain.")
    parser.add_argument('--decision_method', type=str, default="FinalWriteCode", help="Decision method for final node.")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['CodeWriting'], help='Names of agents.')
    parser.add_argument('--num_rounds', type=int, default=1, help="Number of inference rounds.")

    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for training and data generation.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    parser.add_argument('--model_output_dir', type=str, default='output/humaneval_finetuned_model',
                        help="Root output directory for all model artifacts.")

    return parser.parse_args()

def setup_environment(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def get_unique_simple_configs():
    configs = set()
    for agent_num in range(2, 6):
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
    print("\n" + "=" * 20 + " Stage 1: Starting HumanEval pretraining " + "=" * 20)
    pretrain_args = Args().update_args()
    pretrain_args.epochs = args.pretrain_epochs
    pretrain_args.lr = args.pretrain_lr
    pretrain_args.batch_size = args.batch_size
    pretrain_args.seed = args.seed
    pretrain_args.dataset = 'humaneval'
    pretrain_args.pretrain = args.pretrain
    pretrain_args.model_name = 'best_model.pth'
    pretrain_args.data_dir = COLD_START_DIR
    pretrain_args.experiment_path = args.model_output_dir

    graph_dataset, _ = gdata.load_graph_dataset(pretrain_args)
    role_to_id = graph_dataset.role_to_id
    id_to_role = graph_dataset.id_to_role
    num_node_types = len(role_to_id) + 2
    pretrain_args.role_mapping = role_to_id
    pretrain_args.id_to_role = id_to_role
    pretrain_args.START_TOKEN_ID = len(role_to_id)
    pretrain_args.END_TOKEN_ID = len(role_to_id) + 1

    data_statistics = gdata.get_data_statistics(graph_dataset.graph_list)
    data_statistics['num_node_labels'] = num_node_types
    data_statistics['num_edge_labels'] = 1

    correct_graphs = [g for g in graph_dataset if g.graph.get('is_correct')]
    incorrect_graphs = [g for g in graph_dataset if not g.graph.get('is_correct')]
    random.shuffle(correct_graphs)
    random.shuffle(incorrect_graphs)

    train_ratio = 0.9
    correct_train = correct_graphs[:int(len(correct_graphs) * train_ratio)]
    incorrect_train = incorrect_graphs[:int(len(incorrect_graphs) * train_ratio)]
    correct_val = correct_graphs[int(len(correct_graphs) * train_ratio):]
    incorrect_val = incorrect_graphs[int(len(incorrect_graphs) * train_ratio):]

    train_graphs = correct_train + incorrect_train
    val_graphs = correct_val + incorrect_val
    random.shuffle(train_graphs)
    random.shuffle(val_graphs)
    print(f"Train set: {len(train_graphs)}, Validation set: {len(val_graphs)}")

    dataset_train = gdata.GraphListDataset(train_graphs, pretrain_args)
    dataset_validate = gdata.GraphListDataset(val_graphs, pretrain_args)

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=pretrain_args.batch_size, shuffle=True, drop_last=True,
        num_workers=pretrain_args.num_workers, collate_fn=lambda _: _)

    dataloader_validate = torch.utils.data.DataLoader(
        dataset_validate, batch_size=pretrain_args.batch_size, shuffle=False, drop_last=False,
        num_workers=pretrain_args.num_workers, collate_fn=lambda _: _)

    with open(os.path.join(pretrain_args.experiment_path, "configuration.txt"), 'w') as f:
        json.dump(pretrain_args.__dict__, f, indent=2)

    model = ARGDesigner(pretrain_args, data_statistics).to(pretrain_args.device)
    train(pretrain_args, model, dataloader_train, dataloader_validate)

    print("=" * 20 + " Stage 1 complete: Pretraining done " + "=" * 20)
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
                    if len(components) > 1:
                        main_comp = max(components, key=len)
                        for comp in components:
                            if comp != main_comp:
                                graph.add_edge(random.choice(list(comp)), random.choice(list(main_comp)))
        graph_list.append(graph)
    return graph_list

async def generate_pruned_data(args, dataset_subset, model_path, output_dir):
    print("\n" + "=" * 20 + " Stage 2a: Generating pruned graphs " + "=" * 20)
    model = load_model(model_path)
    model.eval()
    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    executor = PyExecutor()
    role_constraints = ROLE_DESCRIPTION

    print(f"Saving pruned graphs to: {output_dir}")
    successful = 0
    total = min(args.limit_generation_samples, len(dataset_subset))
    records = [(i, r) for i, r in enumerate(dataset_subset) if i < total]

    with tqdm(total=total, desc="Pruning data (parallel)") as pbar:
        for i in range(0, len(records), args.batch_size):
            batch = records[i:i + args.batch_size]
            tasks = []
            for idx, rec in batch:
                text = rec["prompt"]
                emb = torch.tensor(sentence_model.encode(text), device=model.args.device).float()
                gen_graph = generate_graph(model, emb, role_constraints, idx)
                eff_graphs = apply_efficiency_strategy(gen_graph, pruning_ratio=args.pruning_ratio)
                if eff_graphs:
                    g = eff_graphs[0]
                    pyg = convert_to_pyg_graph(g, text)
                    tg = TestGraph(domain=args.domain, llm_name=args.llm_name,
                                   decision_method=args.decision_method, pyg_data=pyg)
                    coroutine = tg.arun({"task": text}, num_rounds=1)
                    tasks.append((coroutine, {"pyg": pyg, "rec": rec, "idx": idx, "text": text}))

            if not tasks:
                continue

            results = await asyncio.gather(*(t[0] for t in tasks), return_exceptions=True)
            for result, (_, meta) in zip(results, tasks):
                pbar.update(1)
                if isinstance(result, Exception):
                    continue
                raw = result[0] if isinstance(result, list) and result else result
                code = raw.lstrip("```python\n").rstrip("\n```")
                solved, _, _ = executor.execute(code, [meta['rec']["test"]], timeout=100)
                if solved:
                    successful += 1
                    name = f"eff_q{meta['idx']}_g0_solved.pt"
                    path = os.path.join(output_dir, name)
                    setattr(meta['pyg'], 'is_correct', True)
                    setattr(meta['pyg'], 'question', meta['text'])
                    setattr(meta['pyg'], 'mode', 'EfficientPruned')
                    torch.save(meta['pyg'], path)

    print(f"Stage 2a complete: {successful} pruned graphs saved.")

async def generate_simple_data(args, dataset_subset, output_dir):
    print("\n" + "=" * 20 + " Stage 2b: Generating simple-structure graphs " + "=" * 20)
    configs = get_unique_simple_configs()
    print(f"Generating data for {len(configs)} simple configs...")

    for mode, num in configs:
        print(f"\nProcessing config: Mode={mode}, Agents={num}")
        kwargs = get_kwargs(mode, num)
        roles = random.choices(list(ROLE_DESCRIPTION.keys()), k=num)
        kwargs['node_kwargs'] = [{'role': r} for r in roles]
        graph = Graph(domain=args.domain, llm_name=args.llm_name,
                      agent_names=[args.agent_names[0]] * num,
                      decision_method=args.decision_method, **kwargs)
        await evaluate_and_save_humaneval_simple(graph, dataset_subset, args, mode, num, output_dir)
    print("Stage 2b complete.")

def generate_replay_data(args, output_dir):
    print("\n" + "=" * 20 + " Stage 2c: Generating replay data " + "=" * 20)
    cold_path = COLD_START_DIR
    if not os.path.isdir(cold_path):
        print(f"Error: Missing cold-start data dir '{cold_path}'.")
        return

    files = [f for f in os.listdir(cold_path) if f.endswith(".pt") and ("True" in f or "solved" in f.lower())]
    if not files:
        print(f"Warning: No successful graphs found in '{cold_path}' for replay.")
        return

    n = min(int(len(files) * args.replay_ratio), len(files))
    selected = random.sample(files, n)
    print(f"Copying {len(selected)} files from '{cold_path}' to '{output_dir}'...")
    for fname in tqdm(selected, desc="Copying replay data"):
        shutil.copy(os.path.join(cold_path, fname), os.path.join(output_dir, fname))
    print("Stage 2c complete.")

async def evaluate_and_save_humaneval_simple(graph: Graph, dataset, args, mode: str, num_agents: int, output_dir: str):
    executor = PyExecutor()
    total = 0
    all_recs = list(enumerate(dataset))
    num_batches = math.ceil(len(all_recs) / args.batch_size)

    for batch_idx in tqdm(range(num_batches), desc=f"Eval {mode}-{num_agents}"):
        batch = all_recs[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
        tasks = []
        for idx, rec in batch:
            g_copy = copy.deepcopy(graph)
            inp = {"task": rec["prompt"]}
            pyg = g_copy.to_pyg_graph(inp)
            tg = TestGraph(domain=args.domain, llm_name=args.llm_name,
                           decision_method=args.decision_method, pyg_data=pyg)
            tasks.append((tg.arun(inp, args.num_rounds), {"pyg": pyg, "rec": rec, "idx": idx, "mode": mode, "num": num_agents}))

        results = await asyncio.gather(*(t[0] for t in tasks), return_exceptions=True)
        for res, (_, meta) in zip(results, tasks):
            if isinstance(res, Exception):
                continue
            raw = res[0] if isinstance(res, list) and res else res
            code = raw.lstrip("```python\n").rstrip("\n```")
            solved, _, _ = executor.execute(code, [meta['rec']["test"]], timeout=100)
            if solved:
                total += 1
                name = f"humaneval_{meta['idx']}_{meta['mode']}_{meta['num']}_True.pt"
                path = os.path.join(output_dir, name)
                save_graph_with_features(meta['pyg'], path, {
                    "mode": meta['mode'], "agent_nums": meta['num'],
                    "is_correct": solved, "question": meta['rec']["prompt"]
                })
    print(f"Config {mode}-{num_agents} done. Solved {total}/{len(dataset)} tasks.")

def run_finetuning(model_path, config_path, finetune_data_dir, args):
    print("\n" + "=" * 20 + " Stage 3: Starting finetuning " + "=" * 20)
    finetune_args = Args().update_args()
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    for k, v in cfg.items():
        if k != 'dataset':
            setattr(finetune_args, k, v)
    finetune_args.dataset = 'humaneval'
    finetune_args.data_dir_ef = finetune_data_dir
    finetune_args.data_dir = COLD_START_DIR
    finetune_args.experiment_path = args.model_output_dir
    finetune_args.epochs = args.finetune_epochs
    finetune_args.lr = args.finetune_lr
    finetune_args.batch_size = args.batch_size
    finetune_args.seed = args.seed
    finetune_args.pretrain = False
    finetune_args.model_name = 'ef_best_model.pth'

    eff_dataset, _ = gdata.load_graph_dataset(finetune_args, pretrain=False)
    if not eff_dataset.graph_list:
        print("Warning: No finetune data found; skipping finetuning.")
        return

    stats = gdata.get_data_statistics(eff_dataset.graph_list)
    stats['num_node_labels'] = len(eff_dataset.role_to_id) + 2
    stats['num_edge_labels'] = 1

    ds = gdata.GraphListDataset(eff_dataset.graph_list, finetune_args)
    dl = torch.utils.data.DataLoader(ds, batch_size=finetune_args.batch_size, shuffle=True, collate_fn=lambda x: x)

    model = ARGDesigner(finetune_args, stats).to(finetune_args.device)
    checkpoint = torch.load(os.path.join(model_path, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded pretrained weights; starting finetuning...")
    train(finetune_args, model, dl)
    print("=" * 20 + " Stage 3 complete: Finetuning done " + "=" * 20)


def find_latest_model_dir(base_dir='output', prefix='humaneval_finetuned_model'):
    if not os.path.exists(base_dir):
        return None
    dirs = [d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d)) and d.startswith(prefix)]
    if not dirs:
        return None
    latest = sorted(dirs)[-1]
    return os.path.join(base_dir, latest)


async def main(flag=True):
    cli_args = parse_cli_args()
    cli_args.pretrain = flag
    setup_environment(cli_args)
    loop = asyncio.get_running_loop()

    if cli_args.pretrain:
        ts = time.strftime("%Y%m%d-%H%M%S")
        cli_args.model_output_dir = f"{cli_args.model_output_dir}_{ts}"
        os.makedirs(cli_args.model_output_dir, exist_ok=True)
        print(f"Starting new run; outputs in {cli_args.model_output_dir}")
        pretrained_model_path, config_file_path = await loop.run_in_executor(None, run_pretraining, cli_args)
    else:
        model_dir = cli_args.load_from_dir or find_latest_model_dir()
        if not model_dir or not os.path.exists(model_dir):
            raise FileNotFoundError("Model directory not found. Run with --pretrain or specify --load_from_dir.")
        print(f"Loading model from: {model_dir}")
        cli_args.model_output_dir = model_dir
        pretrained_model_path = model_dir
        config_file_path = os.path.join(model_dir, "configuration.txt")

    os.makedirs(FINETUNE_DATA_DIR, exist_ok=True)
    if not os.path.exists(TASK_SPLIT_FILE):
        raise FileNotFoundError(f"Task split file '{TASK_SPLIT_FILE}' not found.")
    with open(TASK_SPLIT_FILE, 'r') as f:
        task_split = json.load(f)
    indices = task_split['finetune_tasks_indices']
    full = JSONLReader.parse_file("../../datasets/humaneval/humaneval-py.jsonl")
    subset = [full[i] for i in indices]
    print(f"Loaded {len(subset)} tasks for finetune data generation.")

    await generate_pruned_data(cli_args, subset, pretrained_model_path, FINETUNE_DATA_DIR)
    await generate_simple_data(cli_args, subset, FINETUNE_DATA_DIR)
    generate_replay_data(cli_args, FINETUNE_DATA_DIR)

    await loop.run_in_executor(None, run_finetuning, pretrained_model_path, config_file_path, FINETUNE_DATA_DIR, cli_args)
    print("\nAll stages complete. Final model at:", os.path.join(cli_args.model_output_dir))

if __name__ == '__main__':
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main(False))
