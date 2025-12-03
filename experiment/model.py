import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
import os
import pickle
from experiment.utils import precompute_role_embeddings, get_attributes_len
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import networkx as nx
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data._utils.collate import default_collate as collate
import random

EPS = 1e-9

class Graph_to_Adj_Matrix:
    def __init__(self, args, data_statistics):
        self.max_prev_node = args.max_prev_node
        self.max_head_and_tail = args.max_head_and_tail
        self.is_dag = args.is_dag if hasattr(args, 'is_dag') else False

        if self.max_prev_node is None and self.max_head_and_tail is None:
            raise Exception('Please provide max_prev_node or max_head_and_tail')

        self.data_statistics = data_statistics

        len_node_vec, len_edge_vec, num_nodes_to_consider = get_attributes_len(
            data_statistics['num_node_labels'], data_statistics['num_edge_labels'],
            self.max_prev_node, self.max_head_and_tail)

        self.len_node_vec = len_node_vec
        self.len_edge_vec = len_edge_vec
        self.num_nodes_to_consider = num_nodes_to_consider

        self.feature_len = len_node_vec + num_nodes_to_consider * len_edge_vec

    def __call__(self, graph, perm=None):
        x_item = torch.zeros((self.data_statistics["max_num_nodes"], self.feature_len))
        adj_feature_mat = self.graph_to_matrix(graph, perm)
        x_item[0:adj_feature_mat.shape[0], :adj_feature_mat.shape[1]] = adj_feature_mat
        return {'x': x_item, 'len': len(adj_feature_mat)}

    def graph_to_matrix(self, in_graph, perm):
        """Convert graph to adjacency-feature matrix using role labels"""
        len_node_vec = self.len_node_vec
        num_nodes_to_consider = self.num_nodes_to_consider

        n = len(in_graph.nodes())
        order_map = {perm[i]: i for i in range(n)}
        graph = nx.relabel_nodes(in_graph, order_map)

        adj_mat_2d = torch.ones((n, num_nodes_to_consider))
        adj_mat_2d.tril_(diagonal=-1)
        adj_mat_3d = torch.zeros((n, num_nodes_to_consider, self.data_statistics["num_edge_labels"]))

        node_mat = torch.zeros((n, len_node_vec))

        model = getattr(self, 'model', None)

        # Node label processing using role IDs
        for v, data in graph.nodes.data():
            if 'role' in data and model and hasattr(model, 'get_role_id'):
                role_id = model.get_role_id(data['role'])
                node_mat[v, role_id] = 1
            else:
                if model and hasattr(model, 'id_to_role') and model.id_to_role:
                    random_role_id = random.choice(list(model.id_to_role.keys()))
                    node_mat[v, random_role_id] = 1
                else:
                    node_mat[v, 0] = 1

        # Edge processing
        self._process_edges(graph, adj_mat_2d, adj_mat_3d)

        # Combine edge features and binary mask plus two reserved channels
        adj_mat = torch.cat((
            adj_mat_3d,
            adj_mat_2d.unsqueeze(2),
            torch.zeros((n, num_nodes_to_consider, 2))
        ), dim=2)
        adj_mat = adj_mat.view(n, -1)
        return torch.cat((node_mat, adj_mat), dim=1)

    def _process_edges(self, graph, adj_mat_2d, adj_mat_3d):
        """Process edges and update adjacency matrices"""
        for u, v, data in graph.edges.data():
            edge_label = data.get('label', 0)
            if self.max_prev_node is not None:
                if self.is_dag:
                    if u < v and v - u <= self.max_prev_node:
                        adj_mat_3d[v, v - u - 1, edge_label] = 1
                        adj_mat_2d[v, v - u - 1] = 0
                else:
                    if abs(u - v) <= self.max_prev_node:
                        idx = max(u, v)
                        d = abs(u - v) - 1
                        adj_mat_3d[idx, d, edge_label] = 1
                        adj_mat_2d[idx, d] = 0

            elif self.max_head_and_tail is not None:
                if self.is_dag:
                    if u < v:
                        if v - u <= self.max_head_and_tail[1]:
                            adj_mat_3d[v, v - u - 1, edge_label] = 1
                            adj_mat_2d[v, v - u - 1] = 0
                        elif u < self.max_head_and_tail[0]:
                            idx = self.max_head_and_tail[1] + u
                            adj_mat_3d[v, idx, edge_label] = 1
                            adj_mat_2d[v, idx] = 0
                else:
                    idx_uv = abs(u - v) - 1
                    head_idx = self.max_head_and_tail[1] + min(u, v)
                    if abs(u - v) <= self.max_head_and_tail[1]:
                        idx = max(u, v)
                        adj_mat_3d[idx, idx_uv, edge_label] = 1
                        adj_mat_2d[idx, idx_uv] = 0
                    elif min(u, v) < self.max_head_and_tail[0]:
                        idx = max(u, v)
                        adj_mat_3d[idx, head_idx, edge_label] = 1
                        adj_mat_2d[idx, head_idx] = 0


class BFS_Graph_to_Adj_Matrix(Graph_to_Adj_Matrix):
    """Graph processor with BFS ordering and role/constraint embeddings"""

    def __call__(self, graph, perm=None):
        if perm is None:
            if graph.nodes():
                start_node = random.choice(list(graph.nodes()))
                if self.is_dag:
                    try:
                        perm = list(nx.topological_sort(graph))
                    except nx.NetworkXUnfeasible:
                        perm = list(nx.bfs_tree(graph, start_node))
                else:
                    perm = list(nx.bfs_tree(graph, start_node))
                all_nodes = set(graph.nodes())
                remaining = list(all_nodes - set(perm))
                random.shuffle(remaining)
                perm.extend(remaining)
            else:
                perm = []

        result = super().__call__(graph, perm)

        embedding_dim = 384
        role_embeddings = torch.zeros((self.data_statistics["max_num_nodes"], embedding_dim))

        if hasattr(graph, 'role_embeddings') and graph.role_embeddings is not None:
            for i, node_idx in enumerate(perm):
                if node_idx in graph.role_embeddings:
                    role_embeddings[i] = torch.tensor(graph.role_embeddings[node_idx])
        else:
            for i, node_idx in enumerate(perm):
                node_data = graph.nodes[node_idx]
                if 'role_embedding' in node_data:
                    role_embeddings[i] = torch.tensor(node_data['role_embedding'])
                elif 'role' in node_data:
                    print(f"Warning: node {node_idx} has role but no embedding")

        result['role_embeddings'] = role_embeddings
        return result

    def graph_to_matrix(self, in_graph, perm):
        """Convert graph to adjacency-feature matrix with binary edges"""
        len_node_vec = self.len_node_vec
        num_nodes_to_consider = self.num_nodes_to_consider

        n = len(in_graph.nodes())
        order_map = {perm[i]: i for i in range(n)}
        graph = nx.relabel_nodes(in_graph, order_map)

        adj_mat_2d = torch.ones((n, num_nodes_to_consider))
        adj_mat_2d.tril_(diagonal=-1)

        adj_mat_3d = torch.zeros((n, num_nodes_to_consider, 1))

        node_mat = torch.zeros((n, len_node_vec))

        model = getattr(self, 'model', None)

        for v, data in graph.nodes.data():
            if 'role' in data and model and hasattr(model, 'get_role_id'):
                role_id = model.get_role_id(data['role'])
                node_mat[v, role_id] = 1
            else:
                if model and hasattr(model, 'id_to_role') and model.id_to_role:
                    random_role_id = random.choice(list(model.id_to_role.keys()))
                    node_mat[v, random_role_id] = 1
                else:
                    node_mat[v, 0] = 1

        # Process edges with binary labels
        self._process_edges_binary(graph, adj_mat_2d, adj_mat_3d)

        adj_mat = torch.cat((
            adj_mat_3d,
            adj_mat_2d.unsqueeze(2),
            torch.zeros((n, num_nodes_to_consider, 2))
        ), dim=2)
        adj_mat = adj_mat.view(n, -1)
        return torch.cat((node_mat, adj_mat), dim=1)

    def _process_edges_binary(self, graph, adj_mat_2d, adj_mat_3d):
        """Process edges with binary labels (0/1)"""
        for u, v, data in graph.edges.data():
            if self.max_prev_node is not None:
                if self.is_dag:
                    if u < v and v - u <= self.max_prev_node:
                        adj_mat_3d[v, v - u - 1, 0] = 1
                        adj_mat_2d[v, v - u - 1] = 0
                else:
                    idx = abs(u - v) - 1
                    idx_max = max(u, v)
                    adj_mat_3d[idx_max, idx, 0] = 1
                    adj_mat_2d[idx_max, idx] = 0

            elif self.max_head_and_tail is not None:
                if self.is_dag:
                    if u < v:
                        dist = v - u - 1
                        if dist < self.max_head_and_tail[1]:
                            adj_mat_3d[v, dist, 0] = 1
                            adj_mat_2d[v, dist] = 0
                        elif u < self.max_head_and_tail[0]:
                            idx = self.max_head_and_tail[1] + u
                            adj_mat_3d[v, idx, 0] = 1
                            adj_mat_2d[v, idx] = 0
                else:
                    idx = abs(u - v) - 1
                    idx_max = max(u, v)
                    head_idx = self.max_head_and_tail[1] + min(u, v)
                    if idx < self.max_head_and_tail[1]:
                        adj_mat_3d[idx_max, idx, 0] = 1
                        adj_mat_2d[idx_max, idx] = 0
                    elif min(u, v) < self.max_head_and_tail[0]:
                        adj_mat_3d[idx_max, head_idx, 0] = 1
                        adj_mat_2d[idx_max, head_idx] = 0


class MLP_Basic(nn.Module):
    """Basic MLP implementation"""

    def __init__(self, input_size, embedding_size, output_size, dropout=0):
        super(MLP_Basic, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, embedding_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(embedding_size, output_size),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, input):
        return self.mlp(input)


class MLP_Softmax(nn.Module):
    """MLP + Softmax implementation"""

    def __init__(self, input_size, embedding_size, output_size, dropout=0):
        super(MLP_Softmax, self).__init__()
        self.mlp = nn.Sequential(
            MLP_Basic(input_size, embedding_size, output_size, dropout),
            nn.Softmax(dim=-1)
        )

    def forward(self, input):
        return self.mlp(input)


class ARGDesigner(nn.Module):
    def __init__(self, args, data_statistics):
        super(ARGDesigner, self).__init__()
        self.args = args
        self.data_statistics = data_statistics
        embeddings_path = os.path.join(args.data_dir, 'precomputed_role_embeddings.pkl')
        if os.path.exists(embeddings_path):
            with open(embeddings_path, 'rb') as f:
                self.precomputed_embeddings = pickle.load(f)
            print(f"Loaded precomputed embeddings: {len(self.precomputed_embeddings)} entries")
        else:
            print('Precomputed embeddings not found, computing now')
            self.precomputed_embeddings = precompute_role_embeddings(args.dataset, embeddings_path)

        # self.use_role_as_type = getattr(args, 'use_role_as_type', True)
        self.role_to_id = {}
        self.id_to_role = {}

        self.role_to_id = args.role_mapping
        self.id_to_role = {int(k) if isinstance(k, str) else k: v for k, v in args.id_to_role.items()}
        self.START_TOKEN = getattr(args, 'START_TOKEN_ID', len(self.role_to_id))
        self.END_TOKEN = getattr(args, 'END_TOKEN_ID', len(self.role_to_id) + 1)
        print(f"Using provided role mapping: {len(self.role_to_id)} roles")
        print(f"TOKENS: START={self.START_TOKEN}, END={self.END_TOKEN}")

        num_node_types = len(self.role_to_id) + 2
        data_statistics['num_node_labels'] = num_node_types

        len_node_vec, len_edge_vec, num_nodes_to_consider = get_attributes_len(
            data_statistics['num_node_labels'], 1,
            args.max_prev_node, args.max_head_and_tail)

        self.len_node_vec = len_node_vec
        self.len_edge_vec = len_edge_vec
        self.num_nodes_to_consider = num_nodes_to_consider

        self.embedding_dim = getattr(args, 'embedding_dim', 384)

        self.processor = BFS_Graph_to_Adj_Matrix(args, data_statistics)

        feature_len = self.embedding_dim + num_nodes_to_consider * len_edge_vec

        self.task_processor = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )

        self.prev_nodes_aggregator = nn.GRU(
            self.embedding_dim,
            self.embedding_dim,
            batch_first=True
        )

        self.task_node_attention = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1)
        )

        self.node_gru = nn.GRU(
            args.embedding_size_node_level_transformer,
            args.hidden_size_node_level_transformer,
            batch_first=True
        )

        self.edge_gru = nn.GRU(
            args.embedding_size_edge_level_transformer,
            args.hidden_size_edge_level_transformer,
            batch_first=True
        )

        self.node_project = MLP_Basic(
            feature_len,
            args.embedding_size_node_level_transformer,
            args.embedding_size_node_level_transformer
        )

        self.edge_project = MLP_Basic(
            len_edge_vec,
            args.embedding_size_edge_level_transformer,
            args.embedding_size_edge_level_transformer
        )

        self.embedding_node_to_edge = MLP_Basic(
            args.hidden_size_node_level_transformer,
            args.embedding_size_node_level_transformer,
            args.hidden_size_edge_level_transformer
        )

        self.output_node = MLP_Basic(
            args.hidden_size_node_level_transformer,
            args.embedding_size_node_output,
            self.embedding_dim
        )

        self.output_edge = MLP_Softmax(
            args.hidden_size_edge_level_transformer,
            args.embedding_size_edge_level_transformer,
            len_edge_vec
        )

        self.role_processor = MLP_Basic(
            self.embedding_dim,
            self.embedding_dim // 2,
            self.embedding_dim
        )

        role_tensors = []
        for i in range(len(self.id_to_role)):
            role_name = self.id_to_role[i]
            if role_name in self.precomputed_embeddings:
                role_tensors.append(self.precomputed_embeddings[role_name])
            else:
                print(f"Warning: Role '{role_name}' not in embeddings, using random vector")
                role_tensors.append(torch.randn(1, self.embedding_dim))
        base_role_embeddings = torch.stack(role_tensors, dim=0)
        start_embedding = torch.zeros(1, self.embedding_dim)
        end_embedding = torch.zeros(1, self.embedding_dim)
        full_embedding_matrix = torch.cat([base_role_embeddings, start_embedding, end_embedding], dim=0)
        self.register_buffer('full_embedding_matrix', full_embedding_matrix)

    def get_role_id(self, role):
        if role in self.role_to_id:
            return self.role_to_id[role]
        if getattr(self.args, 'debug', False):
            default_role = self.id_to_role.get(0, 'Unknown')
            print(f"Warning: role '{role}' not found, using default '{default_role}'")
        return 0

    def forward(self, batch_graphs, task_embedding=None):
        batch_size = len(batch_graphs)
        role_accuracy = 0.0

        data_batch = []
        for g in batch_graphs:
            if not hasattr(self.processor, 'model'):
                self.processor.model = self
            data = self.processor(g)
            data_batch.append(data)

        data_batch = collate(data_batch)

        x_unsorted = data_batch['x'].to(self.args.device)
        x_len_unsorted = data_batch['len'].to(self.args.device)
        x_len_max = max(x_len_unsorted)

        role_embeddings = data_batch['role_embeddings'].to(self.args.device)
        x_unsorted = x_unsorted[:, :x_len_max, :]
        role_embeddings = role_embeddings[:, :x_len_max, :]

        len_node_vec = self.len_node_vec
        len_edge_vec = self.len_edge_vec
        num_nodes_to_consider = self.num_nodes_to_consider
        feature_len = self.embedding_dim + num_nodes_to_consider * len_edge_vec

        x_len, sort_indices = torch.sort(x_len_unsorted, descending=True)
        x = x_unsorted.index_select(0, sort_indices)
        role_embeddings = role_embeddings.index_select(0, sort_indices)
        if task_embedding is not None:
            task_embedding = task_embedding.index_select(0, sort_indices)
            task_embedding = self.task_processor(task_embedding)

        prev_nodes_embeddings = torch.zeros_like(role_embeddings)
        for b in range(batch_size):
            for i in range(1, x_len[b].item()):
                prev_embs = role_embeddings[b, :i, :]
                if prev_embs.size(0) > 0:
                    _, h_agg = self.prev_nodes_aggregator(prev_embs.unsqueeze(0))
                    h_node = h_agg.squeeze(0).squeeze(0)
                    t_emb = task_embedding[b] if task_embedding is not None else torch.zeros_like(h_node)
                    gate = torch.sigmoid(torch.sum(h_node * t_emb) / self.embedding_dim)
                    if hasattr(self, 'ablation_variant') and self.ablation_variant:
                        if self.ablation_variant == 'no_task':
                            gate = torch.zeros_like(gate)
                        elif self.ablation_variant == 'no_history':
                            gate = torch.ones_like(gate)
                    combined_emb = (1 - gate) * h_node + gate * t_emb
                    prev_nodes_embeddings[b, i, :] = combined_emb

        x_new = torch.zeros(batch_size, x.size(1), feature_len, device=self.args.device)
        x_new[:, :, :self.embedding_dim] = prev_nodes_embeddings
        if x.size(2) > len_node_vec:
            x_new[:, :, self.embedding_dim:] = x[:, :, len_node_vec:]

        node_level_input = torch.cat((
            torch.zeros(batch_size, 1, feature_len, device=self.args.device),
            x_new
        ), dim=1)
        if task_embedding is not None:
            node_level_input[:, 0, :self.embedding_dim] = task_embedding
        node_level_input[:, 0, self.embedding_dim + len_edge_vec - 2] = 1

        h_node = torch.zeros(1, batch_size, self.args.hidden_size_node_level_transformer, device=self.args.device)
        node_level_input = self.node_project(node_level_input)
        node_level_output, h_node = self.node_gru(node_level_input, h_node)

        x_pred_node = self.output_node(node_level_output)
        role_embedding = self.role_processor(self.full_embedding_matrix)
        x_pred_node = torch.matmul(x_pred_node, role_embedding.t())
        x_pred_node = torch.softmax(x_pred_node, dim=-1)
        x_len = x_len.cpu()
        edge_mat_packed = pack_padded_sequence(
            x[:, :, len_node_vec: min(x_len_max - 1, num_nodes_to_consider) * len_edge_vec + len_node_vec],
            x_len, batch_first=True)
        edge_mat, edge_batch_size = edge_mat_packed.data, edge_mat_packed.batch_sizes

        idx = torch.arange(edge_mat.size(0)-1, -1, -1, device=self.args.device)
        edge_mat = edge_mat.index_select(0, idx)
        edge_mat = edge_mat.view(edge_mat.size(0), min(x_len_max - 1, num_nodes_to_consider), len_edge_vec)
        edge_level_input = torch.cat((
            torch.zeros(sum(x_len), 1, len_edge_vec, device=self.args.device),
            edge_mat
        ), dim=1)
        edge_level_input[:, 0, len_edge_vec - 2] = 1

        x_edge_len = []
        x_edge_len_bin = torch.bincount(x_len)
        for i in range(len(x_edge_len_bin) - 1, 0, -1):
            count_temp = torch.sum(x_edge_len_bin[i:]).item()
            x_edge_len.extend([min(i, num_nodes_to_consider + 1)] * count_temp)
        x_edge_len = torch.tensor(x_edge_len, device=self.args.device)

        hidden_edge = self.embedding_node_to_edge(node_level_output[:, :-1, :])
        hidden_edge = pack_padded_sequence(hidden_edge, x_len, batch_first=True).data
        hidden_edge = hidden_edge.index_select(0, idx)
        x_edge_len = x_edge_len.cpu()
        h_edge = hidden_edge.unsqueeze(0)

        edge_level_input = self.edge_project(edge_level_input)
        edge_level_output, _ = self.edge_gru(edge_level_input, h_edge)

        x_pred_edge = self.output_edge(edge_level_output)

        x_pred_node = pack_padded_sequence(x_pred_node, x_len + 1, batch_first=True)
        x_pred_node, _ = pad_packed_sequence(x_pred_node, batch_first=True)
        x_pred_edge = pack_padded_sequence(x_pred_edge, x_edge_len, batch_first=True)
        x_pred_edge, _ = pad_packed_sequence(x_pred_edge, batch_first=True)

        x_node = torch.cat((
            x[:, :, :len_node_vec],
            torch.zeros(batch_size, 1, len_node_vec, device=self.args.device)
        ), dim=1)
        x_node[torch.arange(batch_size), x_len, len_node_vec - 1] = 1

        x_edge = torch.cat((
            edge_mat,
            torch.zeros(sum(x_len), 1, len_edge_vec, device=self.args.device)
        ), dim=1)
        x_edge[torch.arange(sum(x_len)), x_edge_len - 1, len_edge_vec - 1] = 1

        batch_accuracies = []
        for b in range(batch_size):
            true_node_types, pred_node_types = [], []
            for i in range(x_node.size(1)-1):
                if x_node[b, i, :len_node_vec].sum() > 0:
                    true_node_types.append(int(x_node[b, i, :len_node_vec].argmax()))
            for i in range(min(x_pred_node.size(1), len(true_node_types))):
                if x_pred_node[b, i, :len_node_vec].sum() > 0:
                    pred_node_types.append(int(x_pred_node[b, i, :len_node_vec].argmax()))
            correct = sum(1 for i in range(min(len(true_node_types), len(pred_node_types)))
                          if true_node_types[i] == pred_node_types[i])
            total = min(len(true_node_types), len(pred_node_types))
            if total:
                batch_accuracies.append(correct / total * 100)
        if batch_accuracies:
            role_accuracy = sum(batch_accuracies) / len(batch_accuracies)

        loss_node = F.binary_cross_entropy(x_pred_node, x_node, reduction='none')
        loss_edge = F.binary_cross_entropy(x_pred_edge, x_edge, reduction='none')
        loss_node = loss_node.sum(dim=[1, 2])

        edge_batch_size_cum = torch.cat([torch.tensor([0], device=self.args.device),
                                         torch.cumsum(edge_batch_size, dim=0).to(self.args.device)])
        edge_indices = [(edge_batch_size_cum + shift)[:length]
                        for shift, length in enumerate(x_len)]
        loss_edge = torch.cat([
            loss_edge.index_select(0, indices).sum().view(1)
            for indices in edge_indices
        ])

        alpha = 0.2
        loss = alpha * loss_node + (1 - alpha) * loss_edge
        swapped_loss = torch.empty_like(loss)
        swapped_loss[sort_indices] = loss

        log_probs = -swapped_loss
        return log_probs, role_accuracy

    def sample(self, num_samples=10, batch_size=1, task_embedding=None, question_id=None, vis=False):
        role_embeddings_dict_full = self.precomputed_embeddings
        all_roles = list(role_embeddings_dict_full.keys())
        selected_roles_names = all_roles[:]
        role_embeddings_dict = {r: role_embeddings_dict_full[r] for r in selected_roles_names}

        selected_role_ids = [self.role_to_id[r] for r in selected_roles_names]
        end_embedding = self.full_embedding_matrix[self.END_TOKEN].unsqueeze(0)
        candidate_embs = torch.cat(
            [self.full_embedding_matrix[selected_role_ids], end_embedding], dim=0)
        temp_end_idx = len(selected_roles_names)

        min_num_node = self.data_statistics.get("min_num_nodes", 2)
        max_num_node = self.data_statistics.get("max_num_nodes", 10)

        NO_EDGE_TOKEN = 0
        HAS_EDGE_TOKEN = 1
        feature_len = self.embedding_dim + self.num_nodes_to_consider * self.len_edge_vec

        is_dag = getattr(self.args, 'is_dag', False)

        processed_role_embeddings = {}
        for role, emb in role_embeddings_dict.items():
            t = torch.tensor(emb, device=self.args.device).float()
            processed_role_embeddings[role] = t

        if task_embedding is not None:
            if getattr(self, 'ablation_variant', None) == 'random_first_node':
                most_similar_roles = all_roles
            else:
                sims = {
                    r: F.cosine_similarity(
                        processed_role_embeddings[r].unsqueeze(0),
                        task_embedding.unsqueeze(0), dim=1
                    ).item()
                    for r in processed_role_embeddings
                }
                sorted_roles = sorted(sims.items(), key=lambda x: x[1], reverse=True)
                most_similar_roles = [r for r, _ in sorted_roles[:5]]
        else:
            most_similar_roles = all_roles[:5] if len(all_roles) >= 5 else all_roles

        generated_graphs = []
        for batch_idx in range(num_samples // batch_size):
            x_pred_node = np.zeros((batch_size, max_num_node), dtype=np.int32)
            x_pred_edge = np.zeros((batch_size, max_num_node, self.num_nodes_to_consider), dtype=np.int32)
            real_num_nodes = [max_num_node] * batch_size
            generated_roles = [[] for _ in range(batch_size)]
            sampled_node_ids = []
            node_embeddings = [[] for _ in range(batch_size)]

            h_node = torch.zeros(1, batch_size, self.args.hidden_size_node_level_transformer,
                                 device=self.args.device)
            start_token_input = torch.zeros(batch_size, 1, feature_len, device=self.args.device)
            if task_embedding is not None:
                t_proc = self.task_processor(task_embedding)
                start_token_input[:, 0, :self.embedding_dim] = t_proc
            start_token_input[:, 0, self.embedding_dim + self.len_edge_vec - 2] = 1
            start_token_input = self.node_project(start_token_input)
            _, h_node = self.node_gru(start_token_input, h_node)

            finished_flags = [False] * batch_size

            for i in range(max_num_node):
                current_node_input = torch.zeros(batch_size, 1, feature_len, device=self.args.device)
                if i > 0:
                    for b in range(batch_size):
                        prev_embs = torch.stack(node_embeddings[b], dim=0)
                        _, h_agg = self.prev_nodes_aggregator(prev_embs.unsqueeze(0))
                        h_node_hist = h_agg.squeeze(0).squeeze(0)
                        t_emb = t_proc[b] if task_embedding is not None else torch.zeros_like(h_node_hist)
                        gate = torch.sigmoid(torch.sum(h_node_hist * t_emb) / self.embedding_dim)
                        if getattr(self, 'ablation_variant', None) == 'no_task':
                            gate = torch.zeros_like(gate)
                        elif getattr(self, 'ablation_variant', None) == 'no_history':
                            gate = torch.ones_like(gate)
                        combined = (1 - gate) * h_node_hist + gate * t_emb
                        current_node_input[b, 0, :self.embedding_dim] = combined

                active = [idx for idx, done in enumerate(finished_flags) if not done]
                if not active:
                    break

                active_idx_tensor = torch.tensor(active, device=self.args.device)
                active_input = current_node_input.index_select(0, active_idx_tensor)
                active_h_node = h_node.index_select(1, active_idx_tensor)

                proj_in = self.node_project(active_input)
                active_out, active_h_node = self.node_gru(proj_in, active_h_node)
                h_node[:, active_idx_tensor, :] = active_h_node

                pred_node_emb = self.output_node(active_out)

                proc_cand = self.role_processor(candidate_embs)
                scores = torch.matmul(pred_node_emb.squeeze(1), proc_cand.t())
                probs = F.softmax(scores, dim=-1)
                if i < min_num_node:
                    probs[:, temp_end_idx] = 0
                probs = probs / (probs.sum(dim=1, keepdim=True) + EPS)

                sample_out = torch.multinomial(probs, 1).reshape(-1)
                full_out = torch.full((batch_size,), temp_end_idx, device=self.args.device, dtype=torch.long)
                full_out[active_idx_tensor] = sample_out
                sampled_node_ids.append(full_out.cpu().numpy())

                for idx_b, b in enumerate(active):
                    so = sample_out[idx_b].item()
                    if so != temp_end_idx:
                        cand_idx = so
                        node_type_id = selected_role_ids[cand_idx]
                        selected_role = self.id_to_role[node_type_id]
                        generated_roles[b].append(selected_role)
                        if selected_role in processed_role_embeddings:
                            role_emb = processed_role_embeddings[selected_role]
                        else:
                            role_emb = torch.randn(self.embedding_dim, device=self.args.device).float()
                            processed_role_embeddings[selected_role] = role_emb
                        node_embeddings[b].append(role_emb.clone())
                    else:
                        if real_num_nodes[b] == max_num_node:
                            real_num_nodes[b] = i
                        finished_flags[b] = True

                # Edge generation (binary)
                active_output = self.embedding_node_to_edge(active_out)
                edge_input = torch.zeros(len(active), 1, self.len_edge_vec, device=self.args.device)
                edge_input[:, 0, self.len_edge_vec - 2] = 1
                edge_input = self.edge_project(edge_input)
                h_edge = active_output

                for j in range(min(self.num_nodes_to_consider, i)):
                    if j > 0:
                        edge_input = self.edge_project(edge_input)
                    edge_out, h_edge = self.edge_gru(edge_input, h_edge)
                    edge_pred = self.output_edge(edge_out).view(len(active), self.len_edge_vec)
                    exists = torch.bernoulli(edge_pred[:, HAS_EDGE_TOKEN]).long()
                    next_input = torch.zeros(len(active), 1, self.len_edge_vec, device=self.args.device)
                    next_input[:, 0, exists] = 1
                    edge_input = next_input
                    for idx_active, orig in enumerate(active):
                        x_pred_edge[orig, i, j] = exists[idx_active].item()

                # Ensure at least one edge
                for idx_active, orig in enumerate(active):
                    if i > 0:
                        if not x_pred_edge[orig, i].any():
                            j_forced = np.random.randint(0, min(i, self.num_nodes_to_consider))
                            x_pred_edge[orig, i, j_forced] = HAS_EDGE_TOKEN

            # Build final graphs
            for b in range(batch_size):
                G = nx.DiGraph() if is_dag else nx.Graph()
                n_real = real_num_nodes[b]
                for n in range(n_real):
                    node_type = int(x_pred_node[b, n])
                    if node_type != temp_end_idx:
                        role = generated_roles[b][n]
                        G.add_node(n, label=node_type, role=role)
                        if role in processed_role_embeddings:
                            emb = processed_role_embeddings[role].cpu().numpy()
                            G.nodes[n]['role_embedding'] = emb
                for n in range(n_real):
                    if int(x_pred_node[b, n]) == temp_end_idx:
                        continue
                    for j in range(min(self.num_nodes_to_consider, n)):
                        if x_pred_edge[b, n, j] == HAS_EDGE_TOKEN:
                            u = n - j - 1
                            v = n
                            if 0 <= u < n_real and int(x_pred_node[b, u]) != temp_end_idx:
                                G.add_edge(u, v, label=1)
                if is_dag and not nx.is_directed_acyclic_graph(G):
                    for cycle in nx.simple_cycles(G):
                        if cycle:
                            G.remove_edge(cycle[0], cycle[1])
                if G.number_of_nodes() > 1:
                    comps = (nx.weakly_connected_components(G) if is_dag
                             else nx.connected_components(G))
                    comps = list(comps)
                    if len(comps) > 1:
                        main = max(comps, key=len)
                        for comp in comps:
                            if comp != main:
                                src = next(iter(comp))
                                tgt = next(iter(main))
                                if is_dag:
                                    if src < tgt:
                                        G.add_edge(src, tgt, label=1)
                                    else:
                                        G.add_edge(tgt, src, label=1)
                                else:
                                    G.add_edge(src, tgt, label=1)
                for node in list(G.nodes()):
                    if G.degree(node) == 0:
                        others = [n for n in G.nodes() if n != node]
                        if others:
                            tgt = np.random.choice(others)
                            if is_dag:
                                if node < tgt:
                                    G.add_edge(node, tgt, label=1)
                                else:
                                    G.add_edge(tgt, node, label=1)
                            else:
                                G.add_edge(node, tgt, label=1)
                G.graph['mode'] = 'Generated'
                G.graph['agent_nums'] = self.data_statistics.get('agent_nums', 1)
                G.graph['is_correct'] = False
                G.graph['roles'] = generated_roles[b][:n_real]
                G.graph['sampled_node_types'] = [int(sampled_node_ids[i][b]) for i in range(n_real)]
                generated_graphs.append(G)

        return generated_graphs
