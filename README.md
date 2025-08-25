# ARG-Designer

Official implementation of ["Assemble Your Crew: Automatic Multi-agent Communication Topology Design via Autoregressive Graph Generation"](https://arxiv.org/abs/2507.18224).

## Overview

Existing multi-agent systems rely on predefined topologies and manual configuration, limiting their adaptability to diverse tasks. We propose **ARG-Designer**, a novel framework that reframes multi-agent system design as a **conditional autoregressive graph generation task**, automatically determining optimal agent composition and communication structures for each specific task. 

**Key advantages:**

🎯 **Dynamic Agent Assembly**: Sequentially 
determines the optimal number of agents needed for 
each task

🔄 **Intelligent Role Selection**: Chooses 
appropriate agent roles from an extensible pool 
based on task requirements  

🌐 **Optimal Communication Design**: Establishes 
the most effective communication links between 
agents

📊 **Task-Conditioned Generation**: Creates 
customized topologies precisely tailored to unique 
task demands

## Project Structure

```
ARG-Designer/
├── experiment/
│   ├── model.py              # Core ARG-Designer algorithm implementation
│   ├── humaneval/            # HumanEval experiments
│   ├── gsm8k/               # GSM8K experiments
│   ├── mmlu/                # MMLU experiments
│   └── utils.py             # Experiment utilities
├── GDesigner/               # Utility tools (refers to GDesigner framework)
│   ├── agents/             # Agent implementations
│   ├── graph/              # Graph structure utilities
│   ├── llm/                # Language model interfaces
│   ├── tools/              # Coding, search, and other tools
│   └── utils/              # Helper utilities
└── datasets/               # Dataset storage
```

## Quick Start

### Add API keys in `template.env` and change its name to `.env`

```bash
BASE_URL = ""  # the BASE_URL of OpenAI LLM backend
API_KEY = ""   # for OpenAI LLM backend
```

### Run ARG-Designer on HumanEval

The complete workflow consists of three main stages:

#### Stage 1: Cold Start Data Generation

Generate initial training data with different graph topologies and agent configurations:

```bash
cd experiment/humaneval
python cold_start_humaneval.py --dataset_json ../../datasets/humaneval/humaneval-py.jsonl --llm_name gpt-4o-mini --batch_size 4 --num_iterations 10
```

This stage:
- Generates reasoning graphs for different complexity levels  
- Splits the dataset into train/finetune/test sets
- Saves successful graphs as training examples

#### Stage 2: Model Training and Fine-tuning

Train the ARG-Designer model using the generated data:

```bash
python finetune_humaneval.py --pretrain --dataset_name humaneval --llm_name gpt-4o-mini --batch_size 4 --num_iterations 10
```

This stage includes:
- **Pre-training**: Train the base model on cold-start data
- **Data Generation**: Generate additional fine-tuning data using the pre-trained model
- **Fine-tuning**: Perform efficient fine-tuning to optimize efficiency

#### Stage 3: Model Evaluation

Evaluate the trained model on the test set:

```bash
python evaluate_humaneval.py --model_path ./output/your_model_path --task_split_path ./task_split_humaneval.json --llm_name gpt-4o-mini --eval_batch_size 32
```

The evaluation will:
- Load the trained model and test data
- Generate solutions using optimized graph structures
- Execute and verify code solutions
- Report accuracy and performance metrics

### Run ARG-Designer on Other Datasets

#### GSM8K (Mathematical Reasoning)

```bash
cd experiment/gsm8k

# Cold start data generation
python cold_start_gsm8k.py --dataset_name gsm8k --llm_name gpt-4o-mini --batch_size 4 --num_iterations 10

# Training and fine-tuning  
python finetune_gsm8k.py --pretrain --dataset_name gsm8k --llm_name gpt-4o-mini --batch_size 4 --num_iterations 10

# Evaluation
python evaluate_gsm8k.py --model_path ./output/your_model_path --eval_batch_size 32
```

#### MMLU (Knowledge Reasoning)

```bash
cd experiment/mmlu

# Cold start data generation
python cold_start_mmlu.py --dataset_name mmlu --llm_name gpt-4o-mini --batch_size 4 --num_iterations 10

# Training and fine-tuning
python finetune_mmlu.py --pretrain --dataset_name mmlu --llm_name gpt-4o-mini --batch_size 4 --num_iterations 10  

# Evaluation
python evaluate_mmlu.py --model_path ./output/your_model_path --dataset_name mmlu --llm_name gpt-4o-mini --eval_batch_size 32
```

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{li2025assemble,
  title={Assemble your crew: Automatic multi-agent communication topology design via autoregressive graph generation},
  author={Li, Shiyuan and Liu, Yixin and Wen, Qingsong and Zhang, Chengqi and Pan, Shirui},
  journal={arXiv preprint arXiv:2507.18224},
  year={2025}
}
```

## Acknowledgments

This code refers to [GPTSwarm](https://github.com/metauto-ai/GPTSwarm) and [GDesigner](https://github.com/yanweiyue/GDesigner).
