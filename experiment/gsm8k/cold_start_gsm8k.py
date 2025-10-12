import os
import json
import math
import asyncio
import copy
import sys
import argparse
import random
import csv
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.stdout.reconfigure(encoding='utf-8')

from mas_framework.graph.graph import Graph, TestGraph
from mas_framework.tools.reader.readers import JSONLReader
from experiment.utils import get_kwargs, save_graph_with_features
from experiment.gsm8k.gsm8k_prompt_set import ROLE_DESCRIPTION
from datasets.gsm8k_dataset import gsm_data_process, gsm_get_predict

OUTPUT_DIR = "../ColdStartData_gsm8k"
TASK_SPLIT_FILE = "./task_split_gsm8k.json"
BASE_RATE = 0.4


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate cold-start data for GSM8K")
    parser.add_argument('--dataset_json', type=str, default="../../datasets/gsm8k/gsm8k.jsonl",
                        help="Path to GSM8K JSONL dataset")
    parser.add_argument('--llm_name', type=str, default="gpt-4o-mini",
                        help="Name of the LLM model to use")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['MathSolver'],
                        help='List of agent names')
    parser.add_argument('--decision_method', type=str, default="FinalRefer",
                        help="Decision method for the final node")
    parser.add_argument('--num_rounds', type=int, default=1,
                        help="Number of inference rounds per query")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size for generation")
    parser.add_argument('--num_iterations', type=int, default=10,
                        help="Number of iterations to define training set size")
    parser.add_argument('--domain', type=str, default="gsm8k",
                        help="Task domain name")
    return parser.parse_args()


def get_unique_complex_configs_gsm8k():
    """
    Return unique 'complex' topology configurations for GSM8K:
    FullConnected and Mesh with agent counts 3-4.
    """
    configs = set()
    for agent_num in range(2, 5):
        if agent_num == 2:
            continue
        elif agent_num == 3:
            configs.add(('FullConnected', 3))
        else:
            configs.add(('FullConnected', agent_num))
            configs.add(('Mesh', agent_num))
    return list(configs)


def write_to_csv_gsm8k(data):
    filename = "records_gsm8k.csv"
    fieldnames = ["dataset", "id", "question", "answer", "mode", "size", "is_correct"]
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(data)


async def evaluate_and_save_gsm8k(
        graph: Graph,
        dataset,
        args,
        current_mode: str,
        current_agent_num: int
):
    """
    Evaluation and save logic for GSM8K dataset.
    """
    num_batches = math.ceil(len(dataset) / args.batch_size)
    total_solved = 0

    for i_batch in tqdm(range(num_batches), desc=f"Processing {current_mode}-{current_agent_num}"):
        batch_records = dataset[i_batch * args.batch_size: (i_batch + 1) * args.batch_size]
        if not batch_records:
            continue

        tasks = []
        for record_idx, record in enumerate(batch_records):
            realized_graph = copy.deepcopy(graph)
            input_dict = {"task": record["task"]}
            flow_graph = realized_graph.to_pyg_graph(input_dict)
            tg = TestGraph(domain=args.domain, llm_name=args.llm_name,
                           decision_method=args.decision_method, pyg_data=flow_graph)

            metadata = {
                "record": record,
                "flow_graph": flow_graph,
                "question": record["task"],
                "record_idx": i_batch * args.batch_size + record_idx
            }
            tasks.append((tg.arun(input_dict, args.num_rounds), metadata))

        coroutines_to_run = [task for task, _ in tasks]
        results = await asyncio.gather(*coroutines_to_run, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Task execution error: {result}")
                continue

            metadata = tasks[i][1]
            record = metadata['record']

            raw_answer = result
            if isinstance(raw_answer, list) and raw_answer:
                raw_answer = raw_answer[0]

            predict_answer = gsm_get_predict(raw_answer)
            true_answer = record["answer"]
            is_solved = False
            try:
                is_solved = float(predict_answer) == float(true_answer)
            except (ValueError, TypeError):
                print(f"Could not compare answers: predicted='{predict_answer}', true='{true_answer}'")

            if is_solved:
                total_solved += 1
                record_id = metadata['record_idx']
                name = "_".join(map(str, ['gsm8k', record_id, current_mode, current_agent_num, is_solved]))
                filepath = os.path.join(OUTPUT_DIR, f'{name}.pt')
                save_graph_with_features(metadata['flow_graph'], filepath, {
                    "mode": current_mode,
                    "agent_nums": current_agent_num,
                    "is_correct": is_solved,
                    "question": metadata['question']
                })

            batch_data = [{
                "dataset": "gsm8k",
                "id": metadata['record_idx'],
                "question": metadata['question'],
                "answer": true_answer,
                "mode": current_mode,
                "size": current_agent_num,
                "is_correct": is_solved
            }]
            write_to_csv_gsm8k(batch_data)

    print(f"Configuration {current_mode}-{current_agent_num} done. Solved {total_solved}/{len(dataset)} tasks.")


async def main():
    args = parse_args()
    raw_dataset = JSONLReader.parse_file(args.dataset_json)
    dataset = gsm_data_process(raw_dataset)

    train_set_size = args.num_iterations * args.batch_size
    all_indices = list(range(len(dataset)))
    train_indices = all_indices[:train_set_size]
    test_indices = all_indices[train_set_size:]

    print(f"Dataset split: {len(train_indices)} train, {len(test_indices)} test")

    finetune_candidates = train_indices.copy()
    random.shuffle(finetune_candidates)

    base_task_count = int(BASE_RATE * len(finetune_candidates))
    base_task_indices = finetune_candidates[:base_task_count]
    finetune_task_indices = finetune_candidates[base_task_count:]

    with open(TASK_SPLIT_FILE, 'w') as f:
        json.dump({
            "base_tasks_indices": base_task_indices,
            "finetune_tasks_indices": finetune_task_indices,
            "test_indices": test_indices
        }, f)
    print(f"Saved task split: {TASK_SPLIT_FILE} ({len(base_task_indices)} for cold start)")

    cold_start_dataset = [dataset[i] for i in base_task_indices]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    configs = get_unique_complex_configs_gsm8k()
    print(f"Generating GSM8K cold-start data for {len(configs)} configurations...")

    for mode, agent_num in configs:
        print(f"\n=== Configuration: Mode={mode}, Agent Nums={agent_num} ===")
        kwargs = get_kwargs(mode, agent_num)

        available_roles = list(ROLE_DESCRIPTION.keys())
        random_roles = random.choices(available_roles, k=agent_num)
        kwargs['node_kwargs'] = [{'role': role} for role in random_roles]
        print(f"Assigned roles: {random_roles}")

        graph = Graph(domain=args.domain,
                      llm_name=args.llm_name,
                      agent_names=[args.agent_names[0]] * agent_num,
                      decision_method=args.decision_method,
                      **kwargs)

        await evaluate_and_save_gsm8k(
            graph=graph,
            dataset=cold_start_dataset,
            args=args,
            current_mode=mode,
            current_agent_num=agent_num
        )

    print("All GSM8K cold-start data generation complete.")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
