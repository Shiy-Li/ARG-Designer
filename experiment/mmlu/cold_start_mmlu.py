import json
import math
from typing import Optional, Iterator, Any, List
from tqdm import tqdm
import copy
import sys
import os
import csv
import torch
from experiment.utils import get_kwargs, save_graph_with_features, Accuracy
from experiment.mmlu.mmlu_prompt_set import ROLE_DESCRIPTION

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

import asyncio
import argparse
import random
from datasets.mmlu_dataset import MMLUDataset
from datasets.MMLU.download import download
from GDesigner.graph.graph import Graph, TestGraph


def parse_args():
    parser = argparse.ArgumentParser(description="Process parameters for cold-start data generation.")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size for evaluation")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['AnalyzeAgent'],
                        help='List of agent names')
    parser.add_argument('--num_iterations', type=int, default=10,
                        help="Number of optimization iterations")
    parser.add_argument('--num_rounds', type=int, default=1,
                        help="Number of inference rounds for each query")
    parser.add_argument('--llm_name', type=str, default="gpt-4o-mini",
                        help="LLM model name")
    parser.add_argument('--domain', type=str, default="mmlu",
                        help="Domain name, same as dataset name")
    parser.add_argument('--decision_method', type=str, default="FinalRefer",
                        help="Decision method for the final node")
    args = parser.parse_args()

    if len(args.agent_names) != 1:
        parser.error("The number of agent names must be 1.")

    return args


OUTPUT_DIR = "../ColdStartData_mmlu"
TASK_SPLIT_FILE = "./task_split_mmlu.json"
BASE_RATE = 0.4


def get_unique_complex_configs():
    """
    Return unique 'complex' topology configurations: FullConnected and Mesh.
    """
    configs = set()
    for agent_num in range(2, 7):
        if agent_num == 2:
            continue
        elif agent_num == 3:
            configs.add(('FullConnected', 3))
        else:
            configs.add(('FullConnected', agent_num))
            configs.add(('Mesh', agent_num))
    return list(configs)


async def main():
    args = parse_args()
    download()  # Ensure dataset is available

    # Split dataset into base and finetune tasks
    train_set_size = args.num_iterations * args.batch_size
    dataset = MMLUDataset('dev')
    all_indices = list(range(len(dataset)))
    train_indices = all_indices[:train_set_size]
    finetune_candidates = list(train_indices)
    random.shuffle(finetune_candidates)
    base_task_count = int(BASE_RATE * len(finetune_candidates))
    base_task_indices = finetune_candidates[:base_task_count]
    finetune_task_indices = finetune_candidates[base_task_count:]

    # Save task split
    with open(TASK_SPLIT_FILE, 'w') as f:
        json.dump({
            "base_tasks": base_task_indices,
            "finetune_tasks": finetune_task_indices
        }, f)

    print(f"Selected {len(base_task_indices)} base tasks and saved split to {TASK_SPLIT_FILE}")

    # Create cold-start subset
    cold_start_dataset = torch.utils.data.Subset(dataset, base_task_indices)

    # Generate data for each complex configuration
    configs = get_unique_complex_configs()
    print(f"Generating cold-start data for {len(configs)} complex configurations...")

    for mode, agent_num in configs:
        print(f"\n=== Processing configuration: Mode={mode}, Agent Nums={agent_num} ===")

        # Build Graph instance for this configuration
        current_agent_names = [args.agent_names[0]] * agent_num
        kwargs = get_kwargs(mode, agent_num)
        available_roles = list(ROLE_DESCRIPTION.keys())
        random_roles = random.choices(available_roles, k=agent_num)
        kwargs['node_kwargs'] = [{'role': role} for role in random_roles]
        graph = Graph(
            domain=args.domain,
            llm_name=args.llm_name,
            agent_names=current_agent_names,
            decision_method=args.decision_method,
            **kwargs
        )

        # Evaluate and save data
        await evaluate(
            graph=graph,
            dataset=cold_start_dataset,
            num_rounds=args.num_rounds,
            limit_questions=None,
            eval_batch_size=args.batch_size,
            args=args,
            current_mode=mode,
            current_agent_num=agent_num
        )

    print("All cold-start data generation complete.")


async def evaluate(
        graph: Graph,
        dataset,  # Subset of MMLUDataset
        num_rounds: int = 1,
        limit_questions: Optional[int] = None,
        eval_batch_size: int = 1,
        args=None,
        current_mode: str = None,
        current_agent_num: int = None
) -> float:
    """
    Run multi-agent inference on a dataset subset and save successful graphs.
    """
    accuracy = Accuracy()
    original_dataset = dataset.dataset

    def eval_loader(batch_size: int) -> Iterator[List[Any]]:
        records = []
        for i_record, record in enumerate(dataset):
            if limit_questions is not None and i_record >= limit_questions:
                break
            records.append(record)
            if len(records) >= batch_size:
                yield records
                records = []
        if records:
            yield records

    data_len = len(dataset) if limit_questions is None else min(len(dataset), limit_questions)
    num_batches = math.ceil(data_len / eval_batch_size)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for i_batch, record_batch in tqdm(enumerate(eval_loader(batch_size=eval_batch_size)), total=num_batches):
        tasks = []
        questions = []
        flow_graphs = []

        for record in record_batch:
            g_copy = copy.deepcopy(graph)
            input_dict = original_dataset.record_to_input(record)
            flow_graph = g_copy.to_pyg_graph(input_dict)
            tg = TestGraph(
                domain=args.domain,
                llm_name=args.llm_name,
                decision_method=args.decision_method,
                pyg_data=flow_graph
            )
            tasks.append(asyncio.create_task(tg.arun(input_dict, num_rounds)))
            questions.append(input_dict['task'])
            flow_graphs.append(flow_graph)

        is_corrects = []
        raw_results = await asyncio.gather(*tasks)

        for raw_answer, record in zip(raw_results, record_batch):
            answer = original_dataset.postprocess_answer(raw_answer)
            correct_answer = original_dataset.record_to_target_answer(record)
            is_correct = accuracy.update(answer, correct_answer)
            accuracy.print()
            is_corrects.append(is_correct)

        for i, record in enumerate(record_batch):
            record_id = record.get('id', f"task_{i_batch * eval_batch_size + i}")
            batch_data = [{
                "dataset": "mmlu",
                "id": record_id,
                "question": questions[i],
                "mode": current_mode,
                "size": current_agent_num,
                "is_correct": is_corrects[i]
            }]
            write_to_csv(batch_data)
            if is_corrects[i]:
                name = "_".join(map(str, ['mmlu', record_id, current_mode, current_agent_num, is_corrects[i]]))
                filepath = os.path.join(OUTPUT_DIR, f'{name}.pt')
                save_graph_with_features(
                    flow_graphs[i],
                    filepath,
                    {
                        "mode": current_mode,
                        "agent_nums": current_agent_num,
                        "is_correct": is_corrects[i],
                        "question": questions[i]
                    }
                )

    accuracy.print()
    print(f"Finished Mode={current_mode}, Agent Nums={current_agent_num}, Accuracy: {accuracy.get():.2f}%")
    return accuracy.get()


def write_to_csv(data):
    filename = "records.csv"
    fieldnames = ["dataset", "id", "question", "mode", "size", "is_correct"]
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(data)


if __name__ == "__main__":
    asyncio.run(main())
