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
from experiment.utils import save_graph_with_features, get_kwargs, load_model, generate_graph, convert_to_pyg_graph
from sentence_transformers import SentenceTransformer
from experiment import process_dataset as gdata
from mas_framework.graph.graph import Graph, TestGraph
from mas_framework.tools.reader.readers import JSONLReader
from experiments.cold_start import get_kwargs
from datasets.aqua_dataset import aqua_data_process, aqua_get_predict
from experiment.aqua.aqua_prompt_set import ROLE_DESCRIPTION

FINETUNE_DATA_DIR = "/root/ARG-Designer/experiment/FinetuneData_Aqua"
COLD_START_DIR = "/root/ARG-Designer/experiment/ColdStartData_Aqua"
TASK_SPLIT_FILE = "./task_split_aqua.json"


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Full pipeline for Aqua: pretrain, generate data, finetune")
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--load_from_dir', type=str, default=None)

    parser.add_argument('--pretrain_epochs', type=int, default=100)
    parser.add_argument('--pretrain_lr', type=float, default=5e-5)

    parser.add_argument('--pruning_ratio', type=float, default=0.25)
    parser.add_argument('--replay_ratio', type=float, default=0.3)

    parser.add_argument('--finetune_epochs', type=int, default=100)
    parser.add_argument('--finetune_lr', type=float, default=5e-5)

    parser.add_argument('--llm_name', type=str, default="gpt-4o")
    parser.add_argument('--domain', type=str, default="aqua")
    parser.add_argument('--decision_method', type=str, default="FinalRefer")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['MathSolver'])
    parser.add_argument('--num_rounds', type=int, default=1)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model_output_dir', type=str, default='output/aqua_finetuned_model')
    return parser.parse_args()


def setup_environment(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_unique_simple_configs():
    configs = set()
    for agent_num in range(2, 5):
        if agent_num == 2:
            configs.add(('Chain', 2))
        elif agent_num == 3:
            configs.add(('Chain', 3)); configs.add(('Star', 3))
        else:
            configs.add(('Chain', agent_num)); configs.add(('Star', agent_num)); configs.add(('Layered', agent_num))
    return list(configs)


def run_pretraining(args):
    print("\n" + "=" * 20 + " Stage 1: Pretraining Aqua Model " + "=" * 20)
    pretrain_args = Args()
    pretrain_args = pretrain_args.update_args()
    pretrain_args.dataset = 'aqua'
    pretrain_args.data_dir = COLD_START_DIR
    pretrain_args.experiment_path = args.model_output_dir
    pretrain_args.pretrain = args.pretrain
    pretrain_args.epochs = args.pretrain_epochs
    pretrain_args.lr = args.pretrain_lr
    pretrain_args.batch_size = args.batch_size
    pretrain_args.seed = args.seed
    pretrain_args.model_name = 'best_model.pth'

    graph_dataset, _ = gdata.load_graph_dataset(pretrain_args)

    role_to_id, id_to_role = graph_dataset.role_to_id, graph_dataset.id_to_role
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
    train_graphs = correct_graphs[:int(len(correct_graphs) * train_ratio)] + incorrect_graphs[
                                                                            :int(len(incorrect_graphs) * train_ratio)]
    val_graphs = correct_graphs[int(len(correct_graphs) * train_ratio):] + incorrect_graphs[
                                                                           int(len(incorrect_graphs) * train_ratio):]
    random.shuffle(train_graphs)
    random.shuffle(val_graphs)

    dataset_train = gdata.GraphListDataset(train_graphs, pretrain_args)
    dataset_validate = gdata.GraphListDataset(val_graphs, pretrain_args)

    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=pretrain_args.batch_size, shuffle=True,
                                                   drop_last=True, num_workers=0, collate_fn=lambda _: _)
    dataloader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size=pretrain_args.batch_size,
                                                      shuffle=False, drop_last=False, num_workers=0,
                                                      collate_fn=lambda _: _)

    with open(os.path.join(pretrain_args.experiment_path, "configuration.txt"), 'w') as f:
        json.dump(pretrain_args.__dict__, f, indent=2)
    simple_ar_model = SimpleARModel(pretrain_args, data_statistics).to(pretrain_args.device)
    train(pretrain_args, simple_ar_model, dataloader_train, dataloader_validate)

    print("=" * 20 + " Stage 1 Complete " + "=" * 20)
    return os.path.join(pretrain_args.experiment_path), os.path.join(
        pretrain_args.experiment_path, 'configuration.txt')


def apply_efficiency_strategy(graphs, strategy='prune', **kwargs):
    graph_list = []
    for graph in graphs:
        if strategy == 'prune':
            pruning_ratio = kwargs.get('pruning_ratio', 0.2)
            num_edges_to_remove = int(graph.number_of_edges() * pruning_ratio)
            if num_edges_to_remove > 0:
                edges = list(graph.edges())
                random.shuffle(edges)
                graph.remove_edges_from(edges[:num_edges_to_remove])
                if not nx.is_weakly_connected(graph):
                    components = list(nx.weakly_connected_components(graph))
                    if len(components) > 1:
                        main_component = max(components, key=len)
                        for comp in components:
                            if comp != main_component:
                                graph.add_edge(random.choice(list(comp)), random.choice(list(main_component)))
        graph_list.append(graph)
    return graph_list


async def generate_pruned_data(args, dataset_subset, model_path, output_dir):
    print("\n" + "=" * 20 + " Stage 2a: Aqua Pruned Data " + "=" * 20)
    simple_ar_model = load_model(model_path)
    simple_ar_model.eval()
    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    role_constraints_dict = ROLE_DESCRIPTION

    successful_efficient_graphs = 0
    total_samples = min(args.limit_generation_samples, len(dataset_subset))
    all_records_with_indices = [(i, record) for i, record in enumerate(dataset_subset) if i < total_samples]

    with tqdm(total=total_samples, desc="Pruning") as pbar:
        for i in range(0, len(all_records_with_indices), args.batch_size):
            batch_records_with_indices = all_records_with_indices[i:i + args.batch_size]
            batch_tasks = []

            for record_idx, record in batch_records_with_indices:
                task_text = record["task"]
                task_embedding = torch.tensor(sentence_model.encode(task_text),
                                              device=simple_ar_model.args.device).float()
                generated_graph = generate_graph(simple_ar_model, task_embedding, role_constraints_dict,
                                                         record_idx)
                efficient_graphs = apply_efficiency_strategy(generated_graph, strategy='prune',
                                                             pruning_ratio=args.pruning_ratio)

                if efficient_graphs:
                    efficient_graph = efficient_graphs[0]
                    pyg_data = convert_to_pyg_graph(efficient_graph, task_text)
                    tg = TestGraph(domain=args.domain, llm_name=args.llm_name, decision_method=args.decision_method,
                                   pyg_data=pyg_data)
                    coroutine = tg.arun({"task": task_text}, num_rounds=1)
                    metadata = {"pyg_data": pyg_data, "record": record, "task_text": task_text,
                                "record_idx": record_idx}
                    batch_tasks.append((coroutine, metadata))

            if not batch_tasks: continue
            results = await asyncio.gather(*[t[0] for t in batch_tasks], return_exceptions=True)

            for result, (coroutine, metadata) in zip(results, batch_tasks):
                pbar.update(1)
                if isinstance(result, Exception): continue

                raw_answer = result[0] if isinstance(result, (list, tuple)) and result else result
                predict_answer = aqua_get_predict(raw_answer)
                true_answer = metadata['record']["answer"]
                is_solved = False
                try:
                    is_solved = predict_answer == true_answer
                except (ValueError, TypeError):
                    pass

                if is_solved:
                    successful_efficient_graphs += 1
                    file_name = f"eff_q{metadata['record_idx']}_g0_solved.pt"
                    torch.save(metadata['pyg_data'], os.path.join(output_dir, file_name))
    print(f"Stage 2a done, {count} graphs saved")


async def generate_simple_data(args, dataset_subset, output_dir):
    print("\n" + "=" * 20 + " Stage 2b: Aqua Simple Data " + "=" * 20)
    configs = get_unique_simple_configs()
    for mode, agent_num in configs:
        print(f"Config {mode}-{agent_num}")
        kwargs = get_kwargs(mode, agent_num)
        available_roles = list(ROLE_DESCRIPTION.keys())
        kwargs['node_kwargs'] = [{'role': role} for role in random.choices(available_roles, k=agent_num)]
        graph = Graph(domain=args.domain, llm_name=args.llm_name, agent_names=[args.agent_names[0]] * agent_num,
                      decision_method=args.decision_method, **kwargs)
        await evaluate_and_save_aqua_simple(graph, dataset_subset, args, mode, agent_num, output_dir)
    print("Stage 2b done")


def generate_replay_data(args, output_dir):
    print("\n" + "=" * 20 + " Stage 2c: Aqua Replay Data " + "=" * 20)
    cold_start_path = COLD_START_DIR
    replay_files = [f for f in os.listdir(cold_start_path) if
                    f.endswith(".pt") and ("True" in f or "solved" in f.lower())]
    num_to_sample = min(int(len(replay_files) * args.replay_ratio), len(replay_files))
    files_to_copy = random.sample(replay_files, num_to_sample)
    for filename in tqdm(files_to_copy, desc="Copying"):
        shutil.copy(os.path.join(cold_start_path, filename), os.path.join(output_dir, filename))
    print("Stage 2c done")


async def evaluate_and_save_aqua_simple(graph: Graph, dataset, args, current_mode: str, current_agent_num: int,
                                         output_dir: str):
    total_solved = 0
    all_records_with_indices = list(enumerate(dataset))
    num_batches = math.ceil(len(all_records_with_indices) / args.batch_size)

    for i_batch in tqdm(range(num_batches), desc=f"Processing {current_mode}-{current_agent_num}"):
        record_batch_with_indices = all_records_with_indices[i_batch * args.batch_size: (i_batch + 1) * args.batch_size]
        if not record_batch_with_indices: continue

        tasks = []
        for record_idx, record in record_batch_with_indices:
            realized_graph = copy.deepcopy(graph)
            input_dict = {"task": record["task"]}
            flow_graph = realized_graph.to_pyg_graph(input_dict)
            tg = TestGraph(domain=args.domain, llm_name=args.llm_name, decision_method=args.decision_method,
                           pyg_data=flow_graph)
            metadata = {"record": record, "flow_graph": flow_graph, "question": input_dict['task'],
                        "record_idx": record_idx}
            tasks.append((tg.arun(input_dict, args.num_rounds), metadata))

        results = await asyncio.gather(*[t[0] for t in tasks], return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception): continue

            metadata = tasks[i][1]
            raw_answer = result[0] if isinstance(result, (list, tuple)) and result else result
            predict_answer = aqua_get_predict(raw_answer)
            true_answer = metadata['record']["answer"]
            is_solved = False
            try:
                is_solved = predict_answer == true_answer
            except (ValueError, TypeError):
                pass

            if is_solved:
                total_solved += 1
                name = "_".join(map(str, ['aqua', metadata['record_idx'], current_mode, current_agent_num, 'True']))
                filepath = os.path.join(output_dir, f'{name}.pt')
                from experiments.cold_start import save_graph_with_features
                save_graph_with_features(metadata['flow_graph'], filepath,
                                         {"mode": current_mode, "agent_nums": current_agent_num,
                                          "is_correct": is_solved, "question": metadata['question']})

    print(f"Config {current_mode}-{current_agent_num} done, {total_solved} / {len(dataset)} solved")


def run_finetuning(model_path, config_path, finetune_data_dir, args):
    print("\n" + "=" * 20 + " Stage 3: Aqua Finetuning " + "=" * 20)
    finetune_args = Args()
    finetune_args = finetune_args.update_args()
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    for key, value in config_data.items():
        if key not in ['dataset']: setattr(finetune_args, key, value)
    finetune_args.dataset = 'aqua'
    finetune_args.data_dir_ef = finetune_data_dir
    finetune_args.data_dir = COLD_START_DIR
    finetune_args.experiment_path = args.model_output_dir
    finetune_args.pretrain = False
    finetune_args.model_name = 'ef_best_model.pth'
    efficient_dataset, _ = gdata.load_graph_dataset(finetune_args, pretrain=False)
    role_to_id = efficient_dataset.role_to_id
    num_node_types = len(role_to_id) + 2
    if not efficient_dataset.graph_list:
        print("Warning: No finetune data found, skipping finetuning.")
        return

    data_statistics = gdata.get_data_statistics(efficient_dataset.graph_list)
    data_statistics['num_node_labels'] = num_node_types
    data_statistics['num_edge_labels'] = 1
    dataset_finetune = gdata.GraphListDataset(efficient_dataset.graph_list, finetune_args)
    dataloader_finetune = torch.utils.data.DataLoader(dataset_finetune, batch_size=finetune_args.batch_size,
                                                      shuffle=True, collate_fn=lambda x: x)

    simple_ar_model = SimpleARModel(finetune_args, data_statistics).to(finetune_args.device)
    checkpoint = torch.load(os.path.join(model_path, 'best_model.pth'))
    simple_ar_model.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded pretrained weights, starting finetuning...")
    train(finetune_args, simple_ar_model, dataloader_finetune)
    print("=" * 20 + " Stage 3: Aqua Finetuning Complete " + "=" * 20)


def find_latest_model_dir(base_dir='output', prefix='aqua_finetuned_model'):
    if not os.path.exists(base_dir): return None
    candidate_dirs = [d for d in os.listdir(base_dir) if
                      os.path.isdir(os.path.join(base_dir, d)) and d.startswith(prefix)]
    return os.path.join(base_dir, sorted(candidate_dirs)[-1]) if candidate_dirs else None


async def main():
    cli_args = parse_cli_args()
    setup_environment(cli_args.seed)
    loop = asyncio.get_running_loop()

    if cli_args.pretrain:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        cli_args.model_output_dir = f"{cli_args.model_output_dir}_{timestamp}"
        os.makedirs(cli_args.model_output_dir, exist_ok=True)
        print(f"Starting new GSM8K run, outputs to: {cli_args.model_output_dir}")
        pretrained_model_path, config_file_path = await loop.run_in_executor(None, run_pretraining, cli_args)
    else:
        model_dir_to_load = cli_args.load_from_dir or find_latest_model_dir()
        if not model_dir_to_load or not os.path.exists(model_dir_to_load):
            raise FileNotFoundError("No model directory found. Run with --pretrain or specify --load_from_dir.")
        print(f"Loading model from existing directory: {model_dir_to_load}")
        cli_args.model_output_dir = model_dir_to_load
        pretrained_model_path = os.path.join(model_dir_to_load)
        config_file_path = os.path.join(model_dir_to_load, "configuration.txt")
        if not os.path.exists(pretrained_model_path) or not os.path.exists(config_file_path):
            raise FileNotFoundError(f"Missing best_model.pth or configuration.txt in directory  {model_dir_to_load}.")

    finetune_data_output_dir = FINETUNE_DATA_DIR
    os.makedirs(finetune_data_output_dir, exist_ok=True)

    if not os.path.exists(TASK_SPLIT_FILE):
        raise FileNotFoundError(f"Task split file '{TASK_SPLIT_FILE}' not found. Please run cold_start_aqua.py first.")
    with open(TASK_SPLIT_FILE, 'r') as f:
        task_split = json.load(f)

    full_dataset = aqua_data_process(JSONLReader.parse_file("../../datasets/AQuA/AQuA.jsonl"))
    finetune_dataset_subset = [full_dataset[i] for i in task_split['finetune_tasks_indices']]
    print(f"\nLoaded {len(finetune_dataset_subset)} tasks (from training set) for Aqua finetune data generation.")

    await generate_pruned_data(cli_args, finetune_dataset_subset, pretrained_model_path, finetune_data_output_dir)
    await generate_simple_data(cli_args, finetune_dataset_subset, finetune_data_output_dir)
    generate_replay_data(cli_args, finetune_data_output_dir)
    await loop.run_in_executor(None, run_finetuning, pretrained_model_path, config_file_path, finetune_data_output_dir,
                               cli_args)
    print(f"\nAqua pipeline complete! Final efficient model at: {os.path.join(cli_args.model_output_dir)}")


if __name__ == '__main__':
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
