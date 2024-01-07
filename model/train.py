import configparser
import os

import torch
import engine
import model_builder

""" Basic setups """
# Read configs
config = configparser.ConfigParser()
config.read('config.ini')

# Setup hyperparameters
epochs = config.getint('model', 'epochs')
learning_rate = config.getfloat('model', 'learning_rate')

# Setup device and environment
device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ['TORCH'] = torch.__version__

# Setup dataset
data_base_path = config.get('dataset', 'imdb_1m_path')
data_ii = torch.load(f'{data_base_path}/data_ii.pt').to(device)
data_ui_train = torch.load(f'{data_base_path}/data_ui_train.pt').to(device)
data_ui_test = torch.load(f'{data_base_path}/data_ui_test.pt').to(device)
data_ui_validation = torch.load(f'{data_base_path}/data_ui_validation.pt').to(device)

# Setup device and environment
device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ['TORCH'] = torch.__version__

if __name__ == '__main__':
    num_item_nodes, num_features = data_ii.x.size()

    # ItemItem graph
    model = model_builder.BigraphModel(
        channels_ii=[num_features, num_features, num_features],
        channels_ui=[num_features, num_features, num_features],
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_fn = torch.nn.CrossEntropyLoss()

    # Train
    engine.train(
        model=model,
        data_ii=data_ii,
        data_ui_train=data_ui_train,
        data_ui_test=data_ui_test,
        data_ui_validation=data_ui_validation,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=epochs,
        device=device,
    )

    # sample_tensor_normalized = torch.nn.functional.normalize(out_ui, p=2, dim=1)
    # cosine_similarities = torch.matmul(sample_tensor_normalized, sample_tensor_normalized.t())
    # print(torch.min(cosine_similarities), torch.mean(cosine_similarities), torch.max(cosine_similarities))
