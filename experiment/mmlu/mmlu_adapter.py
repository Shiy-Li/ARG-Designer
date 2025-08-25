import os
import glob
import torch
import networkx as nx
import pickle
import random

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer
from mmlu_prompt_set import ROLE_DESCRIPTION


def get_sentence_embedding(sentence):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(sentence)
    return embeddings


def precompute_role_embeddings(save_path="./prompt/precomputed_role_embeddings.pkl"):
    """
    Precompute embeddings for roles defined in ROLE_DESCRIPTION.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    role_embeddings = {}

    for role, description in ROLE_DESCRIPTION.items():
        embedding = model.encode(f"{role}: {description.strip()}")
        role_embeddings[role] = torch.tensor(embedding)

    with open(save_path, 'wb') as f:
        pickle.dump(role_embeddings, f)

    print(f"Precomputed {len(role_embeddings)} role embeddings, saved to {save_path}")
    return role_embeddings


class MMLUGraphDataset:
    """
    Adapter for MMLU graph data, converting PyG graphs to NetworkX DAGs.
    """

    def __init__(self, data_dir, sample_size=0, pretrain=True, data_dir_ef=None):
        self.data_dir = data_dir
        self.data_dir_ef = data_dir_ef
        self.pretrain = pretrain
        cache_path = os.path.join(self.data_dir, 'precomputed_role_embeddings.pkl')
        if not os.path.exists(cache_path):
            self.precomputed_embeddings = precompute_role_embeddings(cache_path)
        else:
            with open(cache_path, 'rb') as f:
                self.precomputed_embeddings = pickle.load(f)
            print(f"Loaded {len(self.precomputed_embeddings)} precomputed embeddings")
        self.graph_list = self._load_and_convert_graphs(sample_size)
        self.node_label_list = [0]
        self.edge_label_list = [0]

    def _load_and_convert_graphs(self, sample_size):
        """
        Load PyG graphs from disk, filter and sample, convert to NetworkX DAGs.
        """
        if self.pretrain:
            graph_files = glob.glob(os.path.join(self.data_dir, '*.pt'))
        else:
            graph_files = glob.glob(os.path.join(self.data_dir_ef, '*.pt'))

        if self.pretrain:
            graph_files = [f for f in graph_files if "True" in os.path.basename(f)]

        if not graph_files:
            print("Warning: No graph files found")
            return []

        if sample_size and sample_size > 0 and len(graph_files) > sample_size:
            random.seed(42)
            graph_files = random.sample(graph_files, sample_size)
        else:
            print(f"Using all {len(graph_files)} graph files")

        # Create global role mapping
        sorted_roles = sorted(ROLE_DESCRIPTION.keys())
        role_to_id = {role: i for i, role in enumerate(sorted_roles)}
        id_to_role = {int(i): role for i, role in enumerate(sorted_roles)}
        self.role_to_id = role_to_id
        self.id_to_role = id_to_role

        nx_graphs = []
        for file in graph_files:
            try:
                pyg_graph = torch.load(file)
                num_nodes = pyg_graph.num_nodes
                nx_graph = nx.DiGraph()
                nx_graph.add_nodes_from(range(num_nodes))
                nx_graph.role_embeddings = {}
                task = getattr(pyg_graph, "question", "")

                for i, node_data in enumerate(pyg_graph.x):
                    role = node_data.get('role')
                    if role not in role_to_id:
                        role = id_to_role[0]
                    constraint = node_data.get('constraint', '')
                    embedding = self.precomputed_embeddings[role]

                    nx_graph.nodes[i]['role'] = role
                    if constraint:
                        nx_graph.nodes[i]['constraint'] = constraint
                    nx_graph.nodes[i]['role_id'] = role_to_id.get(role, 0)
                    nx_graph.role_embeddings[i] = embedding
                    nx_graph.nodes[i]['label'] = role_to_id.get(role, 0)
                    feature_dim = 10
                    nx_graph.nodes[i]['feat'] = [1.0] * feature_dim

                if hasattr(pyg_graph, 'edge_index'):
                    edge_index = pyg_graph.edge_index.numpy()
                    edges = [(int(edge_index[0, j]), int(edge_index[1, j])) for j in range(edge_index.shape[1])]
                    nx_graph.add_edges_from(edges)
                    nx.set_edge_attributes(nx_graph, 0, "label")

                if not nx.is_directed_acyclic_graph(nx_graph):
                    try:
                        _ = list(nx.topological_sort(nx_graph))
                    except nx.NetworkXUnfeasible:
                        for u, v in list(nx_graph.edges()):
                            nx_graph.remove_edge(u, v)
                        for i in range(num_nodes - 1):
                            nx_graph.add_edge(i, i + 1, label=0)

                mode = getattr(pyg_graph, 'mode', None)
                if mode is None:
                    parts = os.path.basename(file).split('_')
                    mode = parts[2] if len(parts) > 2 else "Unknown"
                nx_graph.graph['mode'] = mode

                is_correct = getattr(pyg_graph, 'is_correct', None)
                if is_correct is None:
                    is_correct = "True" in os.path.basename(file)
                nx_graph.graph['is_correct'] = bool(is_correct)

                parts = os.path.basename(file).split('_')
                agent_nums = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 1
                nx_graph.graph['agent_nums'] = agent_nums

                task_embedding = get_sentence_embedding(task)
                nx_graph.graph['task_embedding'] = task_embedding
                nx_graph.graph['is_dag'] = True

                nx_graphs.append(nx_graph)

            except Exception as e:
                print(f"Error processing file {file}: {e}")

        dag_count = sum(nx.is_directed_acyclic_graph(g) for g in nx_graphs)
        print(f"DAG check: {dag_count}/{len(nx_graphs)} graphs are DAG")

        return nx_graphs

    def __getitem__(self, index):
        return self.graph_list[index]

    def __len__(self):
        return len(self.graph_list)
