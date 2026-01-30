import os
import json
import torch
import numpy as np
import argparse
import random
import networkx as nx
from tqdm import tqdm
import asyncio
import copy
import sys
import shutil
import time
import math

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
from datasets.gsm8k_dataset import gsm_data_process, gsm_get_predict
from experiment.gsm8k.gsm8k_prompt_set import ROLE_DESCRIPTION

FINETUNE_DATA_DIR = "/root/ARG-Designer/experiment/FinetuneData_gsm8k"
COLD_START_DIR = "/root/ARG-Designer/experiment/ColdStartData_gsm8k"
TASK_SPLIT_FILE = "./task_split_gsm8k.json"


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Full pipeline for GSM8K: pretrain, generate data, finetune")
    parser.add_argument('--pretrain', action='store_true', help="Run pretraining if set")
    parser.add_argument('--load_from_dir', type=str, default=None, help="Directory to load pretrained model from")

    parser.add_argument('--pretrain_epochs', type=int, default=100, help="Number of pretraining epochs")
    parser.add_argument('--pretrain_lr', type=float, default=1e-4, help="Learning rate for pretraining")

    parser.add_argument('--pruning_ratio', type=float, default=0.25, help="Pruning ratio for data generation")
    parser.add_argument('--replay_ratio', type=float, default=0.3, help="Replay ratio from cold-start data")

    parser.add_argument('--finetune_epochs', type=int, default=200, help="Number of finetuning epochs")
    parser.add_argument('--finetune_lr', type=float, default=5e-5, help="Learning rate for finetuning")

    parser.add_argument('--llm_name', type=str, default="gpt-4o-mini", help="LLM model name")
    parser.add_argument('--domain', type=str, default="gsm8k", help="Task domain")
    parser.add_argument('--decision_method', type=str, default="FinalRefer", help="Decision method")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['MathSolver'], help='List of agent names')
    parser.add_argument('--num_rounds', type=int, default=1, help="Number of inference rounds")

    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training and data gen")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--model_output_dir', type=str, default='output/gsm8k_finetuned_model',
                        help="Root directory for model outputs")
    return parser.parse_args()


def setup_environment(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def get_unique_simple_configs():
    configs = set()
    for agent_num in range(2, 5):
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
    print("\n" + "=" * 20 + " Stage 1: Pretraining GSM8K Model " + "=" * 20)
    pre = Args().update_args()
    pre.dataset = 'gsm8k'
    pre.data_dir = COLD_START_DIR
    pre.experiment_path = args.model_output_dir
    pre.pretrain = args.pretrain
    pre.epochs = args.pretrain_epochs
    pre.lr = args.pretrain_lr
    pre.batch_size = args.batch_size
    pre.seed = args.seed
    pre.model_name = 'best_model.pth'

    ds, _ = gdata.load_graph_dataset(pre)

    role_to_id, id_to_role = ds.role_to_id, ds.id_to_role
    num_node_types = len(role_to_id) + 2
    pre.role_mapping = role_to_id
    pre.id_to_role = id_to_role
    pre.START_TOKEN_ID = len(role_to_id)
    pre.END_TOKEN_ID = len(role_to_id) + 1

    stats = gdata.get_data_statistics(ds.graph_list)
    stats['num_node_labels'] = num_node_types
    stats['num_edge_labels'] = 1

    correct = [g for g in ds if g.graph.get('is_correct')]
    incorrect = [g for g in ds if not g.graph.get('is_correct')]
    random.shuffle(correct)
    random.shuffle(incorrect)

    ratio = 0.9
    train_graphs = correct[:int(len(correct) * ratio)] + incorrect[:int(len(incorrect) * ratio)]
    val_graphs = correct[int(len(correct) * ratio):] + incorrect[int(len(incorrect) * ratio):]
    random.shuffle(train_graphs)
    random.shuffle(val_graphs)

    print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}")
    dt = gdata.GraphListDataset(train_graphs, pre)
    dv = gdata.GraphListDataset(val_graphs, pre)

    lt = torch.utils.data.DataLoader(dt, batch_size=pre.batch_size, shuffle=True,
                                     drop_last=True, num_workers=pre.num_workers, collate_fn=lambda x: x)
    lv = torch.utils.data.DataLoader(dv, batch_size=pre.batch_size, shuffle=False,
                                     drop_last=False, num_workers=pre.num_workers, collate_fn=lambda x: x)

    with open(os.path.join(pre.experiment_path, "configuration.txt"), 'w') as f:
        json.dump(pre.__dict__, f, indent=2)

    model = ARGDesigner(pre, stats).to(pre.device)
    train(pre, model, lt, lv)

    print("=" * 20 + " Stage 1 Complete " + "=" * 20)
    return pre.experiment_path, os.path.join(pre.experiment_path, 'configuration.txt')


def apply_efficiency_strategy(graphs, strategy='prune', **kwargs):
    out = []
    for g in graphs:
        if strategy == 'prune':
            r = kwargs.get('pruning_ratio', 0.2)
            n = int(g.number_of_edges() * r)
            if n > 0:
                es = list(g.edges())
                random.shuffle(es)
                g.remove_edges_from(es[:n])
                if not nx.is_weakly_connected(g):
                    comps = list(nx.weakly_connected_components(g))
                    main = max(comps, key=len)
                    for c in comps:
                        if c != main:
                            g.add_edge(random.choice(list(c)), random.choice(list(main)))
        out.append(g)
    return out


async def generate_pruned_data(args, ds, mp, od):
    print("\n" + "=" * 20 + " Stage 2a: GSM8K Pruned Data " + "=" * 20)
    model = load_model(mp)
    model.eval()
    sm = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    roles = ROLE_DESCRIPTION

    count = 0
    total = len(ds)
    recs = list(enumerate(ds))[:total]

    with tqdm(total=total, desc="Pruning") as p:
        for i in range(0, total, args.batch_size):
            batch = recs[i:i+args.batch_size]
            tasks = []
            for idx, rec in batch:
                t = rec["task"]
                emb = torch.tensor(sm.encode(t), device=model.args.device).float()
                gens = generate_graph(model, emb, roles, idx)
                effs = apply_efficiency_strategy(gens, 'prune', pruning_ratio=args.pruning_ratio)
                if effs:
                    e = effs[0]
                    pd = convert_to_pyg_graph(e, t)
                    tg = TestGraph(domain=args.domain, llm_name=args.llm_name,
                                   decision_method=args.decision_method, pyg_data=pd)
                    metadata = {"pyg_data": pd, "record": rec, "task_text": t, "record_idx": idx}
                    tasks.append((tg.arun({"task": t}, num_rounds=1), metadata))
            if not tasks:
                continue
            res = await asyncio.gather(*[t[0] for t in tasks], return_exceptions=True)
            for out, metadata in zip(res, tasks):
                p.update(1)
                if isinstance(out, Exception):
                    continue
                ans = out[0] if isinstance(out, (list, tuple)) and out else out
                pred = gsm_get_predict(ans)
                true = metadata[1]['record']["answer"]
                if pred == true:
                    count += 1
                    name = f"eff_q{metadata[1]['record_idx']}_g0.pt"
                    torch.save(metadata[1]['pyg_data'], os.path.join(od, name))
    print(f"Stage 2a done, {count} graphs saved")


async def generate_simple_data(args, ds, od):
    print("\n" + "=" * 20 + " Stage 2b: GSM8K Simple Data " + "=" * 20)
    configs = get_unique_simple_configs()
    for mode, num in configs:
        print(f"Config {mode}-{num}")
        kw = get_kwargs(mode, num)
        roles = random.choices(list(ROLE_DESCRIPTION.keys()), k=num)
        kw['node_kwargs'] = [{'role': r} for r in roles]
        g = Graph(domain=args.domain, llm_name=args.llm_name,
                  agent_names=[args.agent_names[0]]*num,
                  decision_method=args.decision_method, **kw)
        await evaluate_and_save_gsm8k_simple(g, ds, args, mode, num, od)
    print("Stage 2b done")


def generate_replay_data(args, od):
    print("\n" + "=" * 20 + " Stage 2c: GSM8K Replay Data " + "=" * 20)
    files = [f for f in os.listdir(COLD_START_DIR) if f.endswith(".pt") and ("True" in f or "solved" in f.lower())]
    N = min(int(len(files)*args.replay_ratio), len(files))
    pick = random.sample(files, N)
    for fn in tqdm(pick, desc="Copying"):
        shutil.copy(os.path.join(COLD_START_DIR, fn), os.path.join(od, fn))
    print("Stage 2c done")


async def evaluate_and_save_gsm8k_simple(graph, ds, args, mode, num, od):
    total_solved = 0
    all_records_with_indices = list(enumerate(ds))
    num_batches = math.ceil(len(all_records_with_indices) / args.batch_size)

    for i_batch in tqdm(range(num_batches), desc=f"Processing {mode}-{num}"):
        record_batch_with_indices = all_records_with_indices[i_batch * args.batch_size: (i_batch + 1) * args.batch_size]
        if not record_batch_with_indices:
            continue

        tasks = []
        for record_idx, record in record_batch_with_indices:
            realized_graph = copy.deepcopy(graph)
            input_dict = {"task": record["task"]}
            flow_graph = realized_graph.to_pyg_graph(input_dict)
            tg = TestGraph(domain=args.domain, llm_name=args.llm_name,
                           decision_method=args.decision_method, pyg_data=flow_graph)
            metadata = {"record": record, "flow_graph": flow_graph, "question": input_dict['task'], "record_idx": record_idx}
            tasks.append((tg.arun(input_dict, args.num_rounds), metadata))

        results = await asyncio.gather(*[t[0] for t in tasks], return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                continue
            
            metadata = tasks[i][1]
            raw_answer = result[0] if isinstance(result, (list, tuple)) and result else result
            predict_answer = gsm_get_predict(raw_answer)
            true_answer = metadata['record']["answer"]
            is_solved = False
            try:
                is_solved = float(predict_answer) == float(true_answer)
            except (ValueError, TypeError):
                pass

            if is_solved:
                total_solved += 1
                name = "_".join(map(str, ['gsm8k', metadata['record_idx'], mode, num, 'True']))
                filepath = os.path.join(od, f'{name}.pt')
                save_graph_with_features(metadata['flow_graph'], filepath, {
                    "mode": mode, "agent_nums": num, "is_correct": is_solved, "question": metadata['question']
                })

    print(f"Config {mode}-{num} done, {total_solved}/{len(ds)} solved")


def run_finetuning(model_path, config_path, finetune_data_dir, args):
    print("\n" + "=" * 20 + " Stage 3: GSM8K Finetuning " + "=" * 20)
    finetune_args = Args()
    finetune_args = finetune_args.update_args()
    finetune_args.ablation = args.ablation
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    for key, value in config_data.items():
        if key not in ['dataset']:
            setattr(finetune_args, key, value)
    finetune_args.dataset = 'gsm8k'
    finetune_args.data_dir_ef = finetune_data_dir
    finetune_args.data_dir = COLD_START_DIR
    finetune_args.experiment_path = args.model_output_dir
    finetune_args.lr = args.finetune_lr
    finetune_args.epochs = args.finetune_epochs
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
    dataloader_finetune = torch.utils.data.DataLoader(dataset_finetune, batch_size=finetune_args.batch_size, shuffle=True, collate_fn=lambda x: x)
    
    model = ARGDesigner(finetune_args, data_statistics).to(finetune_args.device)
    checkpoint = torch.load(os.path.join(model_path, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded pretrained weights, starting finetuning...")
    train(finetune_args, model, dataloader_finetune)
    print("=" * 20 + " Stage 3: GSM8K Finetuning Complete " + "=" * 20)


def find_latest_model_dir(bd='output', pf='gsm8k_finetuned_model'):
    if not os.path.exists(bd): return None
    cds = [d for d in os.listdir(bd) if os.path.isdir(os.path.join(bd, d)) and d.startswith(pf)]
    return os.path.join(bd, sorted(cds)[-1]) if cds else None


async def main():
    args = parse_cli_args()
    setup_environment(args)
    loop = asyncio.get_running_loop()

    if args.pretrain:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        if args.ablation:
            args.model_output_dir = f"{args.model_output_dir}_{args.ablation}_{timestamp}"
        else:
            args.model_output_dir = f"{args.model_output_dir}_{timestamp}"

        os.makedirs(args.model_output_dir, exist_ok=True)
        print(f"Starting new GSM8K run, outputs to: {args.model_output_dir}")
        pretrained_model_path, config_file_path = await loop.run_in_executor(None, run_pretraining, args)
    else:
        model_dir_to_load = args.load_from_dir or find_latest_model_dir()
        if not model_dir_to_load or not os.path.exists(model_dir_to_load):
            raise FileNotFoundError("No model directory found. Run with --pretrain or specify --load_from_dir.")
        print(f"Loading model from existing directory: {model_dir_to_load}")
        args.model_output_dir = model_dir_to_load
        pretrained_model_path = os.path.join(model_dir_to_load)
        config_file_path = os.path.join(model_dir_to_load, "configuration.txt")
        if not os.path.exists(pretrained_model_path) or not os.path.exists(config_file_path):
            raise FileNotFoundError(f"Missing best_model.pth or configuration.txt in directory {model_dir_to_load}.")

    finetune_data_output_dir = FINETUNE_DATA_DIR
    os.makedirs(finetune_data_output_dir, exist_ok=True)

    if not os.path.exists(TASK_SPLIT_FILE):
        raise FileNotFoundError(f"Task split file '{TASK_SPLIT_FILE}' not found. Please run cold_start_gsm8k.py first.")
    with open(TASK_SPLIT_FILE, 'r') as f:
        task_split = json.load(f)
    
    full_dataset = gsm_data_process(JSONLReader.parse_file("../../datasets/gsm8k/gsm8k.jsonl"))
    finetune_dataset_subset = [full_dataset[i] for i in task_split['finetune_tasks_indices']]
    print(f"\nLoaded {len(finetune_dataset_subset)} tasks (from training set) for GSM8K finetune data generation.")

    await generate_pruned_data(args, finetune_dataset_subset, pretrained_model_path, finetune_data_output_dir)
    await generate_simple_data(args, finetune_dataset_subset, finetune_data_output_dir)
    generate_replay_data(args, finetune_data_output_dir)

    await loop.run_in_executor(None, run_finetuning, pretrained_model_path, config_file_path, finetune_data_output_dir, args)
    print(f"\nGSM8K pipeline complete! Final efficient model at: {os.path.join(args.model_output_dir)}")


if __name__ == '__main__':
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
