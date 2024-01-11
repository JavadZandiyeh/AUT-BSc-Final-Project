import configparser
import csv
import sys
from enum import Enum

import numpy as np
import torch
from torch_geometric.data import Dataset, Data


def train_val_test_mask(matrix):
    """ Convert an adjacency matrix to train, val, and test masks
        Args:
            matrix: symmetric adjacency matrix with shape (n * n)

        Returns:
            three matrices contains named: 1. train_mask (80%), 2. val_mask (10%), 3. test_mask (10%)
            these matrices divide our matrix into three parts, which are used for the
            learning procedure.
            example:
            train_mask, val_mask, test_mask = train_val_test_mask(matrix)
    """
    matrix_triu = np.triu(matrix, k=0)  # upper triangular matrix, with the diagonal (k=0)

    num_nonzero = np.count_nonzero(matrix_triu)
    rand_arr = np.zeros(num_nonzero)
    index_1, index_2 = int(0.8 * num_nonzero), int(0.9 * num_nonzero)
    rand_arr[:index_1], rand_arr[index_1:index_2], rand_arr[index_2:] = 1, 2, 3  # train, val, test
    np.random.shuffle(rand_arr)

    rand_index = 0
    row, col = matrix_triu.shape
    for i in range(row):
        for j in range(col):
            if matrix_triu[i][j] != 0:
                matrix_triu[i][j] = rand_arr[rand_index]
                rand_index += 1

    def get_mask(mask_type):
        mask_triu = np.zeros_like(matrix_triu)
        mask_triu[matrix_triu == mask_type] = 1  # with diagonal
        mask_tril = np.triu(mask_triu, k=1).T  # without diagonal
        return mask_triu + mask_tril

    train, val, test = get_mask(1), get_mask(2), get_mask(3)

    return train, val, test


def matrix_sparsify(matrix, degree_cutoff):
    mask = np.where(matrix != 0, 1, 0)

    # Reduce the degree of high-degree nodes
    while True:
        degrees = np.sum(mask, axis=0)
        index = int(np.argmax(degrees))

        if degrees[index] <= degree_cutoff:
            break

        indices = np.argsort(-matrix[index] * mask[index])[:degree_cutoff]

        updated_mask_row = np.zeros_like(mask[index])
        updated_mask_row[indices] = 1

        mask[index, :] = updated_mask_row
        mask[:, index] = updated_mask_row

    # Connect isolate nodes to others
    isolates = np.where(np.all(mask == 0, axis=1))[0]
    for i in isolates:
        indices = np.argsort(-matrix[i])[:len(isolates)]
        mask[i][indices] = 1
        mask[:, i] = mask[i, :]

    return matrix * mask


def matrix_sparsify1(matrix, degree_cutoff):
    base_mask = np.where(matrix != 0, 1, 0)
    final_mask = np.zeros_like(matrix, dtype=np.int64)

    processed = np.ones(matrix.shape[0])

    while not np.all(processed == np.inf):
        prunable_mask = base_mask - final_mask

        degrees = np.sum(prunable_mask, axis=0)
        index = np.argmin(degrees * processed)

        processed[index] = np.inf

        if (num_prunable := degree_cutoff - sum(final_mask[index])) <= 0:
            continue

        indices = np.argsort(-matrix[index] * prunable_mask[index])[:num_prunable]

        final_mask[index][indices] = 1
        final_mask[:, index] = final_mask[index, :]

    return matrix * final_mask


def matrix_sparsify2(matrix, degree_cutoff):
    mask = np.zeros_like(matrix, dtype=np.int64)

    for i in range(matrix.shape[0]):
        indices = np.argsort(-matrix[i])[:degree_cutoff]

        mask[i][indices] = 1
        mask[:, i] = mask[i, :]

    return matrix * mask


class DatasetType(Enum):
    imdb_1m = 'imdb-1m'
    imdb_25m = 'imdb-25m'


class IMDbDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        # Dimension of matrices (because the .dat file doesn't store dimensions of matrices)
        self.num_users = 6040
        if dataset_type == DatasetType.imdb_1m.value:  # imdb-1m
            self.num_items, self.num_item_features = 3848, 31
        else:  # imdb-25m
            self.num_items, self.num_item_features = 62390, 35

        # Initialize matrices
        # self.item_names, self.item_interactions, self.ii_weights, self.user_names = None, None, None, None
        self.user_features, self.item_features, self.item_interactions_diff, self.iu_weights = None, None, None, None

        # The maximum number of connections allowed for each item
        self.max_item_degree_threshold = config.getint('main', 'item_degree')

        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        """ If these files exist in raw_dir, the download is not triggered """
        return ['item_names.csv', 'item_interactions.dat', 'item_interactions_diff.dat', 'item_features.dat',
                'ii_weights.dat', 'user_names.csv', 'user_features.dat', 'iu_weights.dat']

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped """
        return ['data.pt']

    def download(self):
        """ Download raw files if they do not exist in our raw directory """
        pass  # For this dataset, all raw files are already available, and there is no need to download anything.

    def process(self):
        # fetch required matrices to create our graphs
        self.fetch_matrices()

        """ x: [(num_users + num_items), num_item_features] """
        x = np.vstack((self.user_features, self.item_features))

        """ y: [(num_users + num_items), (num_users, num_items)] """
        y_uu = np.zeros((self.num_users, self.num_users))
        y_ii = np.zeros((self.num_items, self.num_items))

        y = np.vstack((np.hstack((y_uu, self.iu_weights.T)), np.hstack((self.iu_weights, y_ii))))

        """ adj: [(num_users + num_items), (num_users, num_items)] """
        adj_uu = np.zeros((self.num_users, self.num_users))
        adj_iu = np.where(self.iu_weights != 0, 1, 0)
        adj_ii = matrix_sparsify(self.item_interactions_diff, self.max_item_degree_threshold)

        adj = np.vstack((np.hstack((adj_uu, adj_iu.T)), np.hstack((adj_iu, adj_ii))))

        """ mask_ii, mask_uiu: [(num_users + num_items), (num_users, num_items)] """
        mask_ii = np.zeros_like(adj)
        mask_ii[self.num_users:, self.num_users:] = 1

        mask_uiu = np.zeros_like(adj)
        mask_uiu[:self.num_users, self.num_users:] = 1
        mask_uiu[self.num_users:, :self.num_users] = 1

        """ mask_train, mask_val, mask_test: [(num_users + num_items), (num_users, num_items)] """
        mask_train_ii, mask_val_ii, mask_test_ii = train_val_test_mask(mask_ii * adj)
        mask_train_uiu, mask_val_uiu, mask_test_uiu = train_val_test_mask(mask_uiu * adj)

        """ save data in the pyg dataset format"""
        edge_index = np.asarray(np.nonzero(adj))

        edge_y = y[edge_index[0], edge_index[1]]

        edge_attr = adj[edge_index[0], edge_index[1]]

        edge_mask_ii = np.where(mask_ii * adj != 0, True, False)
        edge_mask_ii = edge_mask_ii[edge_index[0], edge_index[1]]

        edge_mask_uiu = np.where(mask_uiu * adj != 0, True, False)
        edge_mask_uiu = edge_mask_uiu[edge_index[0], edge_index[1]]

        edge_mask_train = np.where((mask_train_ii + mask_train_uiu) * adj != 0, True, False)
        edge_mask_train = edge_mask_train[edge_index[0], edge_index[1]]

        edge_mask_val = np.where((mask_val_ii + mask_val_uiu) * adj != 0, True, False)
        edge_mask_val = edge_mask_val[edge_index[0], edge_index[1]]

        edge_mask_test = np.where((mask_test_ii + mask_test_uiu) * adj != 0, True, False)
        edge_mask_test = edge_mask_test[edge_index[0], edge_index[1]]

        data = Data(
            x=torch.tensor(x, dtype=torch.float),
            y=torch.tensor(edge_y, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float),
            edge_mask_uiu=torch.tensor(edge_mask_uiu, dtype=torch.bool),
            edge_mask_ii=torch.tensor(edge_mask_ii, dtype=torch.bool),
            edge_mask_train=torch.tensor(edge_mask_train, dtype=torch.bool),
            edge_mask_val=torch.tensor(edge_mask_val, dtype=torch.bool),
            edge_mask_test=torch.tensor(edge_mask_test, dtype=torch.bool)
        )

        torch.save(data, f'{self.processed_dir}/data.pt')

    def fetch_matrices(self):
        """ Fetch data and set our matrices """
        # with open(self.raw_paths[0], 'r', newline='') as csv_file:
        #     csv_reader = csv.reader(csv_file)
        #     self.item_names = [item_name[0] for item_name in csv_reader]

        # self.item_interactions = np.memmap(
        #     self.raw_paths[1],
        #     dtype=np.float32,
        #     mode='r',
        #     shape=(self.num_items, 1)
        # )

        self.item_interactions_diff = np.memmap(
            self.raw_paths[2],
            dtype=np.float32,
            mode='r',
            shape=(self.num_items, self.num_items)
        )

        self.item_features = np.memmap(
            self.raw_paths[3],
            dtype=np.float32,
            mode='r',
            shape=(self.num_items, self.num_item_features)
        )

        # self.ii_weights = np.memmap(
        #     self.raw_paths[4],
        #     dtype=np.float32,
        #     mode='r',
        #     shape=(self.num_items, self.num_items)
        # )

        # with open(self.raw_paths[5], 'r', newline='') as csv_file:
        #     csv_reader = csv.reader(csv_file)
        #     self.user_names = [user_name[0] for user_name in csv_reader]

        self.user_features = np.memmap(
            self.raw_paths[6],
            dtype=np.float32,
            mode='r',
            shape=(self.num_users, self.num_item_features)
        )

        self.iu_weights = np.memmap(
            self.raw_paths[7],
            dtype=np.float32,
            mode='r',
            shape=(self.num_items, self.num_users)
        )

    def len(self):
        return len(self.processed_file_names)

    def get(self, file_name):
        data = torch.load(f'{self.processed_dir}/{file_name}.pt')
        return data


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')

    dataset_type = DatasetType.imdb_1m.value

    dataset = IMDbDataset(root=dataset_type)

    print('data: ', dataset.get('data'))
