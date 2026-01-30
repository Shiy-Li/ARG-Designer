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

from experiment.utils import get_kwargs, save_graph_with_features
from mas_framework.graph.graph import Graph, TestGraph
from mas_framework.tools.reader.readers import JSONLReader
from mas_framework.tools.coding.python_executor import PyExecutor
from experiment.humaneval.humaneval_prompt_set import ROLE_DESCRIPTION

OUTPUT_DIR = "../ColdStartData_humaneval"
TASK_SPLIT_FILE = "./task_split_humaneval.json"
BASE_RATE = 0.4


def parse_args():
    parser = argparse.ArgumentParser(description="Generate cold start data on HumanEval")
    parser.add_argument('--dataset_json', type=str, default="../../datasets/humaneval/humaneval-py.jsonl")
    parser.add_argument('--llm_name', type=str, default="gpt-4o-mini")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['CodeWriting'],
                        help='Specify agent names as a list of strings')
    parser.add_argument('--decision_method', type=str, default="FinalWriteCode",
                        help="Decision method for the final node")
    parser.add_argument('--num_rounds', type=int, default=1, help="Number of inference rounds")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size")
    parser.add_argument('--num_iterations', type=int, default=10,
                        help="Number of iterations to define training set size (consistent with GDesigner)")
    parser.add_argument('--domain', type=str, default="humaneval", help="Task domain")
    return parser.parse_args()


def get_unique_complex_configs_humaneval():
    configs = set()
    for agent_num in range(2, 6):
        if agent_num == 2:
            continue
        elif agent_num == 3:
            configs.add(('FullConnected', 3))
        else:
            configs.add(('FullConnected', agent_num))
            configs.add(('Mesh', agent_num))
    return list(configs)


def write_to_csv_humaneval(data):
    filename = "records_humaneval.csv"
    fieldnames = ["dataset", "id", "question", "mode", "size", "is_correct"]
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(data)


async def evaluate_and_save_humaneval(
        graph: Graph,
        dataset,
        args,
        current_mode: str,
        current_agent_num: int
):
    executor = PyExecutor()
    num_batches = math.ceil(len(dataset) / args.batch_size)
    total_solved = 0

    for i_batch in tqdm(range(num_batches), desc=f"Processing {current_mode}-{current_agent_num}"):
        batch_records = dataset[i_batch * args.batch_size: (i_batch + 1) * args.batch_size]
        if not batch_records:
            continue

        tasks = []
        for record in batch_records:
            realized_graph = copy.deepcopy(graph)
            input_dict = {"task": record["prompt"]}
            flow_graph = realized_graph.to_pyg_graph(input_dict)
            tg = TestGraph(domain=args.domain, llm_name=args.llm_name, decision_method=args.decision_method,
                           pyg_data=flow_graph)

            metadata = {
                "record": record,
                "flow_graph": flow_graph,
                "question": record["prompt"],
            }
            tasks.append((tg.arun(input_dict, args.num_rounds), metadata))

        coroutines_to_run = [task for task, meta in tasks]
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
            answer_code = raw_answer.lstrip("```python\n").rstrip("\n```")

            is_solved, _, _ = executor.execute(answer_code, [record["test"]], timeout=10)

            if is_solved:
                total_solved += 1
                record_id = record.get('task_id', f"task_{i_batch * args.batch_size + i}")
                name = "_".join(map(str, ['humaneval', record_id, current_mode, current_agent_num, is_solved]))
                filepath = os.path.join(OUTPUT_DIR, f'{name}.pt')

                save_graph_with_features(metadata['flow_graph'], filepath, {
                    "mode": current_mode,
                    "agent_nums": current_agent_num,
                    "is_correct": is_solved,
                    "question": metadata['question']
                })

            batch_data = [{
                "dataset": "humaneval",
                "id": record.get('task_id', f"task_{i_batch * args.batch_size + i}"),
                "question": metadata['question'],
                "mode": current_mode,
                "size": current_agent_num,
                "is_correct": is_solved
            }]
            write_to_csv_humaneval(batch_data)

    print(
        f"Config {current_mode}-{current_agent_num} finished. Successfully solved {total_solved} / {len(dataset)} tasks.")


async def main():
    args = parse_args()
    dataset = JSONLReader.parse_file(args.dataset_json)
    train_set_size = args.num_iterations * args.batch_size
    all_indices = list(range(len(dataset)))
    train_indices = all_indices[:train_set_size]
    test_indices = all_indices[train_set_size:]
    finetune_candidates = list(train_indices)
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
    cold_start_dataset = [dataset[i] for i in base_task_indices]
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    configs = get_unique_complex_configs_humaneval()
    for mode, agent_num in configs:
        print("\n" + "=" * 80)
        print(f"Processing config: Mode={mode}, Agent Nums={agent_num}")
        print("=" * 80)
        kwargs = get_kwargs(mode, agent_num)
        available_roles = list(ROLE_DESCRIPTION.keys())
        random_roles = random.choices(available_roles, k=agent_num)
        kwargs['node_kwargs'] = [{'role': role} for role in random_roles]
        graph = Graph(domain=args.domain,
                      llm_name=args.llm_name,
                      agent_names=[args.agent_names[0]] * agent_num,
                      decision_method=args.decision_method,
                      **kwargs)
        await evaluate_and_save_humaneval(
            graph=graph,
            dataset=cold_start_dataset,
            args=args,
            current_mode=mode,
            current_agent_num=agent_num
        )
    print("\nAll HumanEval cold start data generation completed!")


if __name__ == "__main__":
    asyncio.run(main())
