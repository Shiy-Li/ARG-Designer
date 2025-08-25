import datetime
import torch
import argparse
import os


class Args:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('--save_model', default=True, action='store_true', help='Whether to save model')
        self.parser.add_argument('--epochs_validate', type=int, default=1, help='model validate epoch interval')
        self.parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu',
                                 help='cuda:[d] | cpu')
        self.parser.add_argument('--seed', type=int, default=123, help='random seed to reproduce performance/dataset')
        self.parser.add_argument('--dataset', type=str, default='mmlu',
                        help='the name of the dataset. Default: caveman_small')
        self.parser.add_argument('--dataset_name', default='',
                                 help='Name of the dataset (e.g., mmlu)')
        self.parser.add_argument('--data_dir', type=str, default='../../ColdStartData',
                        help='Data directory for storing dataset files, such as MMLU graph data')
        self.parser.add_argument('--hidden_size_node_level_transformer', type=int, default=256,
                                 help='hidden size for node level transformer')
        self.parser.add_argument('--embedding_size_node_level_transformer', type=int, default=256,
                                 help='the size for node level transformer input')
        self.parser.add_argument('--embedding_size_node_output', type=int, default=256,
                                 help='the size of node output embedding')
        self.parser.add_argument('--hidden_size_edge_level_transformer', type=int, default=256,
                                 help='hidden size for edge level transformer')
        self.parser.add_argument('--embedding_size_edge_level_transformer', type=int, default=256,
                                 help='the size for edge level transformer input')

        self.parser.add_argument('--max_prev_node', type=int, default=10)
        self.parser.add_argument('--max_head_and_tail', type=int, default=10)
        self.parser.add_argument('--is_dag', action='store_true', help='Process directed acyclic graphs')
        self.parser.add_argument('--batch_size', type=int, default=2, help='batchsize')
        self.parser.add_argument('--num_workers', type=int, default=1, help='number of workers for dataloader')
        self.parser.add_argument('--epochs', type=int, default=51, help='epochs')

        self.parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
        self.parser.add_argument('--gamma', type=float, default=0.3, help='Learning rate decay factor')
        self.parser.add_argument('--clip', default=True, action='store_true',
                                 help='whether to use clip gradient for generation model')
        self.parser.add_argument('--role_mapping_path', type=str, default=None,
                        help='Path to predefined role mapping file, if provided will load role mapping from here')

    def update_args_from_dict(self, args_dict):
        """
        Update parameters from dictionary.
        """
        for key, value in args_dict.items():
            setattr(self, key, value)

    def update_args(self):
        """
        Update args when load a trained model: use settings from the saved model
        """
        args = self.parser.parse_args()
        args.time = '{0:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now())
        args.is_dag = True
        args.dir_input = 'output/'
        args.dataset_path = 'datasets/'
        args.dataset_name = args.dataset
        if args.role_mapping_path is not None and os.path.exists(args.role_mapping_path):
            try:
                import json
                with open(args.role_mapping_path, 'r') as f:
                    args.role_mapping = json.load(f)
                print(f"Loaded {len(args.role_mapping)} role mappings from {args.role_mapping_path}")
            except Exception as e:
                print(f"Failed to load role mapping: {e}")
        return args
