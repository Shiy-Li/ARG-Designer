import os
import glob
import torch
import networkx as nx
import pickle
import random
from sentence_transformers import SentenceTransformer

from experiment.aqua_prompt_set_adapter import ROLE_DESCRIPTION

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_sentence_embedding(sentence):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(sentence)
    return embeddings


def precompute_role_embeddings(save_path):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    role_embeddings = {}
    for role, description in ROLE_DESCRIPTION.items():
        role_with_desc = f"{role}: {description.strip()}"
        full_embedding = model.encode(role_with_desc)
        role_embeddings[role] = torch.tensor(full_embedding)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(role_embeddings, f)

    print(f"Precomputed {len(role_embeddings)} role embeddings, saved to {save_path}")
    return role_embeddings


class AquaGraphDataset:
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
            print(f"Loaded {len(self.precomputed_embeddings)} precomputed role embeddings")

        self.graph_list = self._load_and_convert_graphs(sample_size)
        self.node_label_list = [0]
        self.edge_label_list = [0]

    def _load_and_convert_graphs(self, sample_size):
        if self.pretrain:
            graph_files = glob.glob(os.path.join(self.data_dir, '*.pt'))
        else:
            if not self.data_dir_ef:
                print("Warning: data_dir_ef not provided")
                return []
            graph_files = glob.glob(os.path.join(self.data_dir_ef, '*.pt'))

        print(f"Found {len(graph_files)} graph files")

        if self.pretrain:
            true_graph_files = [f for f in graph_files if
                                "True" in os.path.basename(f) or "solved" in os.path.basename(f).lower()]
            print(f"Filtered {len(true_graph_files)} successful graphs for pretraining")
            graph_files = true_graph_files

        if not graph_files:
            print("Warning: No valid graph files found")
            return []

        if sample_size and sample_size > 0 and len(graph_files) > sample_size:
            random.seed(42)
            graph_files = random.sample(graph_files, sample_size)
            print(f"Randomly sampled {sample_size} graphs")
        else:
            print(f"Using all {len(graph_files)} graphs")

        nx_graphs = []

        sorted_roles = sorted(list(ROLE_DESCRIPTION.keys()))
        role_to_id = {role: i for i, role in enumerate(sorted_roles)}
        id_to_role = {int(i): role for i, role in enumerate(sorted_roles)}

        print(f"Created unified role mapping with {len(role_to_id)} roles")

        self.role_to_id = role_to_id
        self.id_to_role = id_to_role

        for file in graph_files:
            try:
                try:
                    pyg_graph = torch.load(file, weights_only=False)
                except TypeError:
                    pyg_graph = torch.load(file)
                num_nodes = pyg_graph.num_nodes
                nx_graph = nx.DiGraph()
                nx_graph.add_nodes_from(range(num_nodes))

                nx_graph.role_embeddings = {}
                task = getattr(pyg_graph, "question", "")

                for i, node_data in enumerate(pyg_graph.x):
                    role = node_data.get('role', id_to_role[0])
                    if role not in role_to_id:
                        role = id_to_role[0]

                    embedding = self.precomputed_embeddings.get(role)
                    if embedding is None:
                        embedding = torch.zeros_like(next(iter(self.precomputed_embeddings.values())))

                    nx_graph.nodes[i]['role'] = role
                    nx_graph.nodes[i]['role_id'] = role_to_id[role]
                    nx_graph.nodes[i]['label'] = role_to_id[role]
                    nx_graph.role_embeddings[i] = embedding

                if hasattr(pyg_graph, 'edge_index'):
                    edge_index = pyg_graph.edge_index.numpy()
                    for i in range(edge_index.shape[1]):
                        u, v = int(edge_index[0, i]), int(edge_index[1, i])
                        nx_graph.add_edge(u, v, label=0)

                if not nx.is_directed_acyclic_graph(nx_graph):
                    edges_to_remove = []
                    for cycle in nx.simple_cycles(nx_graph):
                        edges_to_remove.append((cycle[-2], cycle[-1]))
                    nx_graph.remove_edges_from(edges_to_remove)

                nx_graph.graph['mode'] = getattr(pyg_graph, 'mode', 'Unknown')
                nx_graph.graph['is_correct'] = getattr(pyg_graph, 'is_correct', False)
                nx_graph.graph['agent_nums'] = getattr(pyg_graph, 'agent_nums', 1)
                task_embedding = get_sentence_embedding(task)
                nx_graph.graph['task_embedding'] = task_embedding
                nx_graph.graph['is_dag'] = True

                nx_graphs.append(nx_graph)

            except Exception as e:
                print(f"Error processing file {file}: {e}")

        dag_count = sum(1 for g in nx_graphs if nx.is_directed_acyclic_graph(g))
        print(f"DAG check: {dag_count}/{len(nx_graphs)} graphs are DAG")
        return nx_graphs

    def __getitem__(self, index):
        return self.graph_list[index]

    def __len__(self):
        return len(self.graph_list)
