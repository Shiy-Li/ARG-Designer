import os
import json
import torch
import argparse
import asyncio
from tqdm import tqdm
import sys
import datetime
from mas_framework.utils.globals import Cost, PromptTokens, CompletionTokens

os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sentence_transformers import SentenceTransformer
from mas_framework.tools.reader.readers import JSONLReader
from mas_framework.graph.graph import TestGraph
from experiment.utils import load_model, generate_graph, convert_to_pyg_graph
from datasets.aqua_dataset import aqua_data_process, aqua_get_predict
from finetune_aqua import setup_environment
from aqua_prompt_set import ROLE_DESCRIPTION


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model on Aqua")
    parser.add_argument('--model_path', type=str,
                        default='output/xxxx',
                        help="Path to trained model directory")
    parser.add_argument('--dataset_path', type=str, default='../../datasets/AQuA/AQuA.jsonl')
    parser.add_argument('--task_split_path', type=str, default='./task_split_aqua.json')
    parser.add_argument('--llm_name', type=str, default="gpt-4o")
    parser.add_argument('--decision_method', type=str, default="FinalRefer")
    parser.add_argument('--output_file', type=str, default='aqua_eval_results.jsonl')
    parser.add_argument('--summary_log_file', type=str, default='./res_logs/evaluation_summary.jsonl')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--eval_batch_size', type=int, default=10)
    return parser.parse_args()


async def main(ef=True):
    args = parse_args()
    setup_environment(42)

    Cost.instance().reset()
    PromptTokens.instance().reset()
    CompletionTokens.instance().reset()

    print("Loading model and tools...")
    simple_ar_model = load_model(args.model_path, ef=ef)
    if ef:
        args.model_name = 'ef_best'
    else:
        args.model_name = 'best'
    simple_ar_model.eval()
    sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    role_constraints_dict = ROLE_DESCRIPTION

    full_dataset_raw = JSONLReader.parse_file(args.dataset_path)
    full_dataset = aqua_data_process(full_dataset_raw)

    if not os.path.exists(args.task_split_path):
        raise FileNotFoundError(f"Task split file '{args.task_split_path}' not found. Run cold_start_aqua.py first.")
    with open(args.task_split_path, 'r') as f:
        task_split = json.load(f)
    test_indices = task_split.get('test_indices')
    if not test_indices:
        raise ValueError("Task split file does not contain 'test_indices'.")

    dataset = [full_dataset[i] for i in test_indices]

    if args.limit:
        dataset = dataset[:args.limit]
    print(f"Loaded {len(dataset)} Aqua test samples for evaluation.")

    total_tasks = len(dataset)
    solved_tasks = 0
    results_list = []

    from typing import Iterator, List, Any
    import math

    def eval_loader(data: List[Any], batch_size: int) -> Iterator[List[Any]]:
        records = []
        for record in data:
            records.append(record)
            if len(records) >= batch_size:
                yield records
                records = []
        if records:
            yield records
    
    num_batches = math.ceil(total_tasks / args.eval_batch_size)

    pbar = tqdm(enumerate(eval_loader(dataset, args.eval_batch_size)), total=num_batches, desc="Evaluating model")
    for i_batch, record_batch in pbar:
        answer_tasks = []
        metadata_for_tasks = []

        for i_record, record in enumerate(record_batch):
            task_text = record["task"]
            true_answer = record["answer"]
            global_idx = i_batch * args.eval_batch_size + i_record
            task_id = f"task_{test_indices[global_idx]}"

            try:
                task_embedding = torch.tensor(sentence_model.encode(task_text),
                                              device=simple_ar_model.args.device).float()
                generated_graphs = generate_graph(simple_ar_model, task_embedding, role_constraints_dict, global_idx)

                if not generated_graphs:
                    raise RuntimeError("Graph generation failed.")

                generated_graph = generated_graphs[0]
                pyg_data = convert_to_pyg_graph(generated_graph, task_text)
                test_graph = TestGraph(domain="aqua", llm_name=args.llm_name,
                                       decision_method=args.decision_method, pyg_data=pyg_data)

                coro = test_graph.arun({"task": task_text}, num_rounds=1)
                answer_tasks.append(coro)
                metadata_for_tasks.append({
                    "task_id": task_id, "task_text": task_text, "true_answer": true_answer,
                    "generated_graph": generated_graph
                })

            except Exception as e:
                print(f"Error preparing task {task_id}: {e}")
                results_list.append({
                    "task_id": task_id, "question": task_text, "true_answer": true_answer,
                    "predicted_answer": None, "raw_response": None, "is_solved": False, "error": str(e)
                })

        if not answer_tasks:
            continue
        
        all_results = await asyncio.gather(*answer_tasks, return_exceptions=True)

        for i, result in enumerate(all_results):
            metadata = metadata_for_tasks[i]
            task_id = metadata["task_id"]
            task_text = metadata["task_text"]
            true_answer = metadata["true_answer"]
            generated_graph = metadata["generated_graph"]
            
            if isinstance(result, Exception):
                print(f"Error executing task {task_id}: {result}")
                results_list.append({
                    "task_id": task_id, "question": task_text, "true_answer": true_answer,
                    "predicted_answer": None, "raw_response": None, "is_solved": False, "error": str(result)
                })
                continue
                
            raw_answer = result
            raw_answer = raw_answer[0] if isinstance(raw_answer, list) and raw_answer else raw_answer

            predict_answer = aqua_get_predict(raw_answer)
            is_solved = predict_answer == true_answer

            if is_solved:
                solved_tasks += 1
            
            results_list.append({
                "task_id": task_id, "question": task_text, "true_answer": true_answer,
                "predicted_answer": predict_answer, "raw_response": raw_answer, "is_solved": is_solved,
                "num_nodes": generated_graph.number_of_nodes(), "num_edges": generated_graph.number_of_edges(),
            })

        current_processed = len(results_list)
        acc = solved_tasks / current_processed * 100 if current_processed > 0 else 0
        pbar.set_postfix({
            "Accuracy": f"{acc:.2f}% ({solved_tasks}/{current_processed})",
            "Token": f"${PromptTokens.instance().value:.4f}",
        })
        
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for res in results_list:
                f.write(json.dumps(res) + '\n')

    pass_at_1 = (solved_tasks / total_tasks) * 100 if total_tasks > 0 else 0
    final_cost = Cost.instance().value
    final_prompt_tokens = PromptTokens.instance().value
    final_completion_tokens = CompletionTokens.instance().value

    print("\n" + "=" * 50 + "\nEvaluation Summary")
    print(f"Model path: {args.model_path}")
    print(f"Total tasks: {total_tasks}, Solved: {solved_tasks}, Pass@1: {pass_at_1:.2f}%")
    print("-" * 50)
    print(f"Total cost: ${final_cost:.6f}")
    print(f"Total Prompt Tokens: {int(final_prompt_tokens)}")
    print(f"Total Completion Tokens: {int(final_completion_tokens)}")
    print("-" * 50)
    print(f"Detailed results saved to: {args.output_file}")

    log_record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "dataset": "aqua",
        "model_path": args.model_path+args.model_name,
        "llm_name": args.llm_name,
        "total_tasks": total_tasks,
        "solved_tasks": solved_tasks,
        "pass_at_1": pass_at_1,
        "cost": final_cost,
        "prompt_tokens": final_prompt_tokens,
        "completion_tokens": final_completion_tokens,
        "detail_file": args.output_file
    }

    try:
        with open(args.summary_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_record) + '\n')
        print(f"Summary appended to: {args.summary_log_file}")
    except Exception as e:
        print(f"Failed to write summary log file: {e}")
    print("=" * 50)


if __name__ == '__main__':
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main(True))
