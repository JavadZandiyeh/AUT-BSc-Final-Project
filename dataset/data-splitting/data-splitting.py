import csv
import sys
import numpy as np
import torch
from torch_geometric.data import Dataset, Data


def train_test_validation_division(matrix):
    """ Convert an adjacency matrix to train, test, and validation
        Args:
            matrix: adjacency matrix

        Returns:
            three matrices contains named: 1. train (80%), 2. test (10%), 3. validation (10%)
            these matrices divide our matrix into three parts, which are used for the
            learning procedure.
            example:
            train, test, validation = train_test_validation(matrix)
    """

    num_nonzero = np.count_nonzero(matrix)
    rand_arr = np.zeros(num_nonzero)
    index_1, index_2 = int(0.8 * num_nonzero), int(0.9 * num_nonzero)
    rand_arr[:index_1], rand_arr[index_1:index_2], rand_arr[index_2:] = 1, 2, 3  # train, test, validation
    np.random.shuffle(rand_arr)

    rand_index = 0
    ui_adj = np.copy(matrix)  # divide user-item adjacency matrix to 3 parts of (80%, 10%, 10%)
    row, col = matrix.shape
    for i in range(row):
        for j in range(col):
            if ui_adj[i][j] != 0:
                ui_adj[i][j] = rand_arr[rand_index]
                rand_index += 1

    def get_matrix_values(matrix_type):
        mask_matrix = np.zeros_like(matrix)
        mask_matrix[ui_adj == matrix_type] = 1
        return mask_matrix * matrix

    train = get_matrix_values(1)  # matrix_type of 1 is the train matrix
    test = get_matrix_values(2)  # matrix_type of 2 is the test matrix
    validation = get_matrix_values(3)  # matrix_type of 3 is the validation matrix

    return train, test, validation


class IMDbDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        # Initialize matrices
        self.user_names, self.user_features, self.item_names = None, None, None
        self.item_features, self.ii_weights, self.ui_weights = None, None, None

        # Dimension of matrices (because the .dat file doesn't store dimensions of matrices)
        self.num_items, self.num_users, self.num_item_features = 3848, 6040, 31

        # Connections between items with a weight less than this threshold are ignored
        self.ii_weight_threshold = 0.1

        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        """ If these files exist in raw_dir, the download is not triggered """
        return ['user_names.csv', 'user_features.dat', 'item_names.csv', 'item_features.dat', 'ii_weights.dat',
                'ui_weights.dat']  # The order matters in the process() function.

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped """
        return ['data_ii.pt', 'data_ui_train.pt', 'data_ui_test.pt', 'data_ui_validation.pt']

    def download(self):
        """ Download raw files if they do not exist in our raw directory """
        pass  # For this dataset, all raw files are already available, and there is no need to download anything.

    def process(self):
        # Fetch matrices
        self.raw_file_content()

        """ item-item dataset """
        ii_edge_index = np.asarray(np.where(self.ii_weights > self.ii_weight_threshold))
        ii_edge_attr = self.ii_weights[ii_edge_index[0], ii_edge_index[1]]

        data_ii = Data(
            x=torch.tensor(self.item_features),  # node features
            edge_index=torch.tensor(ii_edge_index),  # edges between items
            edge_attr=torch.tensor(ii_edge_attr)  # weights related to edges
        )
        # Example: data_ii = Data(x=[3848, 31], edge_index=[2, 4455450], edge_attr=[4455450])

        torch.save(data_ii, f'{self.processed_dir}/data_ii.pt')

        """ user-item train, test, and validation datasets """
        # part 1
        ui_node_features = np.vstack((self.item_features, self.user_features))

        # part 2
        ui_weights_train, ui_weights_test, ui_weights_validation = train_test_validation_division(self.ui_weights)

        def get_ui_edge_weight_completed(ui_weights):
            ui_edge_index_part1 = np.vstack((ui_weights, np.zeros((self.num_users, self.num_users))))
            ui_edge_index_part2 = np.vstack((np.zeros((self.num_items, self.num_items)), ui_weights.T))
            ui_edge_index = np.hstack((ui_edge_index_part1, ui_edge_index_part2))
            return ui_edge_index

        ui_edge_weights_train = get_ui_edge_weight_completed(ui_weights_train)
        ui_edge_weights_test = get_ui_edge_weight_completed(ui_weights_test)
        ui_edge_weights_validation = get_ui_edge_weight_completed(ui_weights_validation)

        # part 3
        ui_edge_index_train = np.asarray(np.where(ui_edge_weights_train > 0))
        ui_edge_index_test = np.asarray(np.where(ui_edge_weights_test > 0))
        ui_edge_index_validation = np.asarray(np.where(ui_edge_weights_validation > 0))

        # part 4
        ui_edge_attr_train = ui_edge_weights_train[ui_edge_index_train[0], ui_edge_index_train[1]]
        ui_edge_attr_test = ui_edge_weights_test[ui_edge_index_test[0], ui_edge_index_test[1]]
        ui_edge_attr_validation = ui_edge_weights_validation[ui_edge_index_validation[0], ui_edge_index_validation[1]]

        # part 5
        def get_data_ui(ui_edge_index, ui_edge_attr):
            return Data(
                x=torch.tensor(ui_node_features),
                edge_index=torch.tensor(ui_edge_index),
                edge_attr=torch.tensor(ui_edge_attr)
            )

        data_ui_train = get_data_ui(ui_edge_index_train, ui_edge_attr_train)
        data_ui_test = get_data_ui(ui_edge_index_test, ui_edge_attr_test)
        data_ui_validation = get_data_ui(ui_edge_index_validation, ui_edge_attr_validation)

        torch.save(data_ui_train, f'{self.processed_dir}/data_ui_train.pt')
        torch.save(data_ui_test, f'{self.processed_dir}/data_ui_test.pt')
        torch.save(data_ui_validation, f'{self.processed_dir}/data_ui_validation.pt')

        # Examples:
        # data_ui_train = Data(x=[9888, 31], edge_index=[2, 1594986], edge_attr=[1594986])
        # data_ui_test = Data(x=[9888, 31], edge_index=[2, 199374], edge_attr=[199374])
        # data_ui_validation = Data(x=[9888, 31], edge_index=[2, 199374], edge_attr=[199374])

        # Delete matrices to reduce memory usage, as we don't need them anymore
        del self.user_names, self.user_features, self.item_names, self.item_features, self.ii_weights, self.ui_weights

    def raw_file_content(self):
        """ Fetch data and set our matrices """
        with open(self.raw_paths[0], 'r', newline='') as csv_file:
            csv_reader = csv.reader(csv_file)
            self.user_names = [user_name[0] for user_name in csv_reader]

        self.user_features = np.memmap(
            self.raw_paths[1],
            dtype=np.float32,
            mode='r',
            shape=(self.num_users, self.num_item_features)
        )

        with open(self.raw_paths[2], 'r', newline='') as csv_file:
            csv_reader = csv.reader(csv_file)
            self.item_names = [item_name[0] for item_name in csv_reader]

        self.item_features = np.memmap(
            self.raw_paths[3],
            dtype=np.float32,
            mode='r',
            shape=(self.num_items, self.num_item_features)
        )

        # item-item weights
        self.ii_weights = np.memmap(
            self.raw_paths[4],
            dtype=np.float32,
            mode='r',
            shape=(self.num_items, self.num_items)
        )

        # user-item weights
        self.ui_weights = np.memmap(
            self.raw_paths[5],
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
    dataset = IMDbDataset(root=sys.argv[1])
    print(dataset.get('data_ii'))
    print(dataset.get('data_ui_train'))
    print(dataset.get('data_ui_test'))
    print(dataset.get('data_ui_validation'))
