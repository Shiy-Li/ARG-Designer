import math
import sys
import os
import torch
import pickle
import random
import json
import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

def get_kwargs(mode: str, N: int):
    initial_spatial_probability = 0.5
    initial_temporal_probability = 0.5
    fixed_spatial_masks = None
    fixed_temporal_masks = None
    node_kwargs = None

    def generate_layered_graph(N, layer_num=2):
        adj = [[0] * N for _ in range(N)]
        base = N // layer_num
        rem = N % layer_num
        layers = []
        for i in range(layer_num):
            size = base + (1 if i < rem else 0)
            layers.extend([i] * size)
        random.shuffle(layers)
        for i in range(N):
            for j in range(N):
                if layers[j] == layers[i] + 1:
                    adj[i][j] = 1
        return adj

    def generate_mesh_graph(N):
        if N > 4 and int(math.sqrt(N))**2 == N:
            size = int(math.sqrt(N))
            adj = [[0] * N for _ in range(N)]
            for i in range(N):
                if (i + 1) % size != 0:
                    adj[i][i+1] = adj[i+1][i] = 1
                if i < N - size:
                    adj[i][i+size] = adj[i+size][i] = 1
            return adj
        return [[1 if i != j else 0 for i in range(N)] for j in range(N)]

    def generate_star_graph(N):
        adj = [[0] * N for _ in range(N)]
        for i in range(1, N):
            adj[0][i] = adj[i][0] = 1
        return adj

    if mode == 'DirectAnswer':
        fixed_spatial_masks = [[0]]
        fixed_temporal_masks = [[0]]
        node_kwargs = [{'role': 'Normal'}]
    elif mode in ('FullConnected', 'FakeFullConnected', 'FakeAGFull'):
        fixed_spatial_masks = [[1 if i != j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1] * N for _ in range(N)]
    elif mode in ('Random', 'FakeRandom', 'FakeAGRandom'):
        fixed_spatial_masks = [[random.randint(0,1) if i != j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[random.randint(0,1) for _ in range(N)] for _ in range(N)]
    elif mode in ('Chain', 'FakeChain'):
        fixed_spatial_masks = [[1 if abs(i-j)==1 else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 if i==j else 0 for i in range(N)] for j in range(N)]
    elif mode == 'Layered':
        fixed_spatial_masks = generate_layered_graph(N)
        fixed_temporal_masks = [[1]*N for _ in range(N)]
    elif mode in ('Mesh', 'FakeMesh'):
        fixed_spatial_masks = generate_mesh_graph(N)
        fixed_temporal_masks = [[1]*N for _ in range(N)]
    elif mode in ('Star', 'FakeStar'):
        fixed_spatial_masks = generate_star_graph(N)
        fixed_temporal_masks = [[1]*N for _ in range(N)]

    elif 'Fake' in mode and 'AG' not in mode:
        node_kwargs = [{'role': 'Fake'} if i % 2 == N % 2 else {'role': 'Normal'} for i in range(N)]
    elif 'Fake' in mode and 'AG' in mode:
        node_kwargs = [{'role': 'Fake'} if i % 2 == N % 2 else {'role': None} for i in range(N)]

    return {
        "initial_spatial_probability": initial_spatial_probability,
        "fixed_spatial_masks": fixed_spatial_masks,
        "initial_temporal_probability": initial_temporal_probability,
        "fixed_temporal_masks": fixed_temporal_masks,
        "node_kwargs": node_kwargs
    }


def save_graph_with_features(flow_graph, filepath, metadata):
    """
    Attach metadata to the graph and save it.
    """
    for key, value in metadata.items():
        setattr(flow_graph, key, value)
    torch.save(flow_graph, filepath)


def load_model(model_dir, ef=False):
    from experiment.args import Args
    from experiment.model import ARGDesigner

    model_file = os.path.join(model_dir, "ef_best_model.pth" if ef else "best_model.pth")
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"No model file found at {model_file}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_file, map_location=device)
    if not all(k in checkpoint for k in ('args', 'data_statistics', 'model_state_dict')):
        raise ValueError("Invalid checkpoint format. Missing required keys.")

    saved_args = checkpoint['args']
    data_statistics = checkpoint['data_statistics']
    saved_args['data_dir'] = '../ColdStartData_' + saved_args.get('dataset', '')
    args = Args()
    args.update_args_from_dict(saved_args)
    args.device = device

    model = ARGDesigner(args, data_statistics).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def generate_graph(model, task_embedding, role_constraints_dict, question_id=None):
    """
    Generate a graph structure for the given task embedding.
    """
    with torch.no_grad():
        generated = model.sample(num_samples=1, batch_size=1,
                                 task_embedding=task_embedding,
                                 question_id=question_id, vis=True)
    results = []
    for g in generated:
        for n in g.nodes():
            role = g.nodes[n].get('role', 'Unknown')
            g.nodes[n]['constraint'] = role_constraints_dict.get(role, "")
        results.append(g)
    return results


def convert_to_pyg_graph(nx_graph, task_text):
    """
    Convert a NetworkX graph into PyG Data format.
    """
    from torch_geometric.data import Data
    pyg = Data()
    num_nodes = nx_graph.number_of_nodes()
    features = []
    for i in range(num_nodes):
        features.append({
            'role': nx_graph.nodes[i].get('role', 'Unknown'),
            'constraint': nx_graph.nodes[i].get('constraint', '')
        })
    pyg.x = features
    edges = [[u, v] for u, v in nx_graph.edges()]
    pyg.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros((2,0), dtype=torch.long)
    pyg.task = task_text
    pyg.num_nodes = num_nodes
    return pyg


def precompute_role_embeddings(dsets, save_path="./prompt/precomputed_role_embeddings.pkl"):
    """
    Precompute role embeddings for different datasets.
    """
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    role_embeddings = {}

    if dsets == 'mmlu':
        from experiment.mmlu.mmlu_prompt_set import ROLE_DESCRIPTION
    elif dsets == 'humaneval':
        from experiment.humaneval.humaneval_prompt_set import ROLE_DESCRIPTION
    elif dsets == 'svamp':
        from experiment.svamp.svamp_prompt_set import ROLE_DESCRIPTION
    elif dsets == 'aqua':
        from experiment.aqua.aqua_prompt_set import ROLE_DESCRIPTION
    elif dsets == 'gsm8k':
        from experiment.gsm8k.gsm8k_prompt_set import ROLE_DESCRIPTION
    elif dsets == 'multiarith':
        from experiment.multiarith.multiarith_prompt_set import ROLE_DESCRIPTION

    for role, description in ROLE_DESCRIPTION.items():
        full_embedding = model.encode(f"{role}: {description.strip()}")
        role_embeddings[role] = torch.tensor(full_embedding)

    with open(save_path, 'wb') as f:
        pickle.dump(role_embeddings, f)

    print(f"Precomputed {len(role_embeddings)} role embeddings, saved to {save_path}")
    return role_embeddings


class Accuracy:
    """
    Simple accuracy tracker.
    """
    def __init__(self):
        self._num_correct = 0
        self._num_total = 0

    def update(self, predicted: str, target: str) -> bool:
        is_correct = predicted == target
        self._num_correct += int(is_correct)
        self._num_total += 1
        return is_correct

    def get(self) -> float:
        return self._num_correct / self._num_total if self._num_total > 0 else 0.0

    def print(self):
        acc = self.get() * 100
        print(f"Accuracy: {acc:.1f}% ({self._num_correct}/{self._num_total})")

def get_attributes_len(
    len_node_map, len_edge_map, max_prev_node=None, max_head_and_tail=None
):
    """
    Returns (len_node_vec, len_edge_vec, feature_len)
    len_node_vec : Length of vector to represent a node attribute
    len_edge_vec : Length of vector to represent an edge attribute
    num_nodes_to_consider: Number of previous nodes to consider for edges for a given node
    """

    # Last two bits for START node and END node token
    len_node_vec = len_node_map
    # Last three bits in order are NO edge, START egde, END edge token
    len_edge_vec = len_edge_map + 3

    if max_prev_node is not None:
        num_nodes_to_consider = max_prev_node
    elif max_head_and_tail is not None:
        num_nodes_to_consider = max_head_and_tail[0] + max_head_and_tail[1]

    return len_node_vec, len_edge_vec, num_nodes_to_consider
