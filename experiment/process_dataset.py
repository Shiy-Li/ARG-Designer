import torch.utils.data


def load_graph_dataset(args, pretrain=True):
    if args.dataset == 'mmlu':
        from experiment.mmlu.mmlu_adapter import MMLUGraphDataset
        sample_size = getattr(args, 'dataset_sample_size', 0)
        if pretrain:
            dataset = MMLUGraphDataset(data_dir=args.data_dir, sample_size=sample_size, pretrain=pretrain)
        else:
            dataset = MMLUGraphDataset(
                data_dir=args.data_dir,
                sample_size=sample_size,
                pretrain=pretrain,
                data_dir_ef=args.data_dir_ef
            )
        args.max_prev_node = 3
        args.max_head_and_tail = None
        role_embeddings_dict = {}
        for graph in dataset.graph_list:
            for node, data in graph.nodes(data=True):
                if 'role' in data and 'role_embedding' in data:
                    role_embeddings_dict[data['role']] = data['role_embedding']
        return dataset, role_embeddings_dict

    elif args.dataset == 'humaneval':
        from experiment.humaneval.humaneval_adapter import HumanevalGraphDataset
        sample_size = getattr(args, 'dataset_sample_size', 0)
        if pretrain:
            dataset = HumanevalGraphDataset(data_dir=args.data_dir, sample_size=sample_size, pretrain=pretrain)
        else:
            dataset = HumanevalGraphDataset(
                data_dir=args.data_dir,
                sample_size=sample_size,
                pretrain=pretrain,
                data_dir_ef=args.data_dir_ef
            )
        args.max_prev_node = 3
        args.max_head_and_tail = None
        role_embeddings_dict = {}
        if hasattr(dataset, 'precomputed_embeddings'):
            role_embeddings_dict = dataset.precomputed_embeddings
        return dataset, role_embeddings_dict

    elif args.dataset == 'gsm8k':
        from experiment.gsm8k.gsm8k_adapter import Gsm8kGraphDataset
        sample_size = getattr(args, 'dataset_sample_size', 0)
        if pretrain:
            dataset = Gsm8kGraphDataset(data_dir=args.data_dir, sample_size=sample_size, pretrain=pretrain)
        else:
            dataset = Gsm8kGraphDataset(
                data_dir=args.data_dir,
                sample_size=sample_size,
                pretrain=pretrain,
                data_dir_ef=args.data_dir_ef
            )
        args.max_prev_node = 3
        args.max_head_and_tail = None
        role_embeddings_dict = {}
        if hasattr(dataset, 'precomputed_embeddings'):
            role_embeddings_dict = dataset.precomputed_embeddings
        return dataset, role_embeddings_dict

    elif args.dataset == 'svamp':
        from experiment.svamp.svamp_adapter import SvampGraphDataset
        sample_size = getattr(args, 'dataset_sample_size', 0)
        if pretrain:
            dataset = SvampGraphDataset(data_dir=args.data_dir, sample_size=sample_size, pretrain=pretrain)
        else:
            dataset = SvampGraphDataset(
                data_dir=args.data_dir,
                sample_size=sample_size,
                pretrain=pretrain,
                data_dir_ef=args.data_dir_ef
            )
        args.max_prev_node = 3
        args.max_head_and_tail = None
        role_embeddings_dict = {}
        if hasattr(dataset, 'precomputed_embeddings'):
            role_embeddings_dict = dataset.precomputed_embeddings
        return dataset, role_embeddings_dict

    elif args.dataset == 'multiarith':
        from experiment.multiarith.multiarith_adapter import MultiArithGraphDataset
        sample_size = getattr(args, 'dataset_sample_size', 0)
        is_finetune = getattr(args, 'is_finetune', False)
        if not is_finetune:
            dataset = MultiArithGraphDataset(data_dir=args.data_dir, sample_size=sample_size, pretrain=True)
        else:
            dataset = MultiArithGraphDataset(
                data_dir=args.data_dir,
                sample_size=sample_size,
                pretrain=False,
                data_dir_ef=args.data_dir_ef
            )
        args.max_prev_node = 3
        args.max_head_and_tail = None
        role_embeddings_dict = {}
        if hasattr(dataset, 'precomputed_embeddings'):
            role_embeddings_dict = dataset.precomputed_embeddings
        return dataset, role_embeddings_dict

    elif args.dataset == 'aqua':
        from experiment.aqua.aqua_adapter import AquaGraphDataset
        sample_size = getattr(args, 'dataset_sample_size', 0)
        is_finetune = getattr(args, 'is_finetune', False)
        if not is_finetune:
            dataset = AquaGraphDataset(data_dir=args.data_dir, sample_size=sample_size, pretrain=True)
        else:
            dataset = AquaGraphDataset(
                data_dir=args.data_dir,
                sample_size=sample_size,
                pretrain=False,
                data_dir_ef=args.data_dir_ef
            )
        args.max_prev_node = 3
        args.max_head_and_tail = None
        role_embeddings_dict = {}
        if hasattr(dataset, 'precomputed_embeddings'):
            role_embeddings_dict = dataset.precomputed_embeddings
        return dataset, role_embeddings_dict

    else:
        raise Exception(f"Unsupported dataset: {args.dataset}")


class GraphListDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper for a list of networkx graphs.
    """

    def __init__(self, graph_list, args):
        super(GraphListDataset, self).__init__()
        if len(graph_list) == 0:
            print("Warning: Created an empty GraphListDataset")
            self.graph_list = []
            self.node_label_list = None
            self.edge_label_list = None
            return

        self.graph_list = graph_list
        self.args = args
        self.node_label_list = self._map_node_labels()
        self.edge_label_list = self._map_edge_labels()

    def __getitem__(self, index):
        return self.graph_list[index]

    def __len__(self):
        return len(self.graph_list)

    def _map_node_labels(self):
        """
        Map node labels using provided role mapping or default to integer labels.
        """
        if not self.graph_list:
            return None

        role_mapping = getattr(self.args, 'role_mapping', None)
        if not role_mapping:
            # Fallback mapping of existing labels
            label_set = set()
            for graph in self.graph_list:
                for _, data in graph.nodes.data():
                    label_set.add(data.get('label', 0))
            label_list = list(label_set)
            label_dict = {label: idx for idx, label in enumerate(label_list)}
            for graph in self.graph_list:
                for _, data in graph.nodes.data():
                    data['label'] = label_dict[data.get('label', 0)]
            return label_list

        # Use provided role-to-ID mapping
        print("Updating node labels using unified role mapping")
        for graph in self.graph_list:
            for _, data in graph.nodes.data():
                if 'role' in data:
                    data['label'] = role_mapping.get(data['role'], 0)
        return list(role_mapping.keys())

    def _map_edge_labels(self):
        """
        Map edge labels to integers based on first graph's labels.
        """
        if not self.graph_list:
            return None

        first_graph = self.graph_list[0]
        if first_graph.number_of_edges() == 0:
            print("Dataset contains no edges")
            return None

        first_edge = next(iter(first_graph.edges(data=True)))
        if "label" not in first_edge[2]:
            print("Dataset contains no edge labels")
            return None

        label_set = set()
        for graph in self.graph_list:
            for _, _, data in graph.edges.data():
                label_set.add(data['label'])
        label_list = list(label_set)
        label_dict = {label: idx for idx, label in enumerate(label_list)}
        for graph in self.graph_list:
            for _, _, data in graph.edges.data():
                data['label'] = label_dict[data['label']]
        return label_list


def get_data_statistics(dataset):
    """
    Compute data statistics for a dataset or list of graphs.
    """
    if isinstance(dataset, list):
        if len(dataset) == 0:
            print("Warning: get_data_statistics received an empty list, returning default stats")
            return {
                "num_node_labels": 1,
                "num_edge_labels": 1,
                "max_num_nodes": 0,
                "min_num_nodes": 0,
                "max_num_edges": 0,
                "min_num_edges": 0,
                "is_directed": True
            }
        # Gather stats across list of graphs
        node_labels = set()
        edge_labels = set()
        for graph in dataset:
            for _, data in graph.nodes.data():
                node_labels.add(data.get('label', 0))
            for _, _, data in graph.edges.data():
                edge_labels.add(data.get('label', 0))
        stats = {
            "num_node_labels": len(node_labels) or 1,
            "num_edge_labels": len(edge_labels) or 1,
            "max_num_nodes": max(g.number_of_nodes() for g in dataset),
            "min_num_nodes": min(g.number_of_nodes() for g in dataset),
            "max_num_edges": max(g.number_of_edges() for g in dataset),
            "min_num_edges": min(g.number_of_edges() for g in dataset),
            "is_directed": dataset[0].is_directed()
        }
        return stats

    # Handle GraphListDataset or similar
    if len(dataset) == 0:
        print("Warning: get_data_statistics received an empty dataset, returning default stats")
        return {
            "num_node_labels": 1,
            "num_edge_labels": 1,
            "max_num_nodes": 0,
            "min_num_nodes": 0,
            "max_num_edges": 0,
            "min_num_edges": 0,
            "is_directed": True
        }

    stats = {
        "num_node_labels": len(dataset.node_label_list) or 1,
        "num_edge_labels": len(dataset.edge_label_list) or 1,
    }
    graph = dataset[0]
    stats.update({
        "max_num_nodes": graph.number_of_nodes(),
        "min_num_nodes": graph.number_of_nodes(),
        "max_num_edges": graph.number_of_edges(),
        "min_num_edges": graph.number_of_edges(),
        "is_directed": graph.is_directed()
    })
    for item in dataset:
        g = item['workflow'] if isinstance(item, dict) else item
        stats["max_num_nodes"] = max(stats["max_num_nodes"], g.number_of_nodes())
        stats["min_num_nodes"] = min(stats["min_num_nodes"], g.number_of_nodes())
        stats["max_num_edges"] = max(stats["max_num_edges"], g.number_of_edges())
        stats["min_num_edges"] = min(stats["min_num_edges"], g.number_of_edges())
    return stats