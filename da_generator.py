import numpy as np

from torch import nn
import torch

from sysidentpy.neural_network import NARXNN
from sysidentpy.basis_function import Polynomial
from sysidentpy.utils.information_matrix import count_model_regressors

from datasets import CWRU

def train(net, x_train, y_train, xlag, ylag, y_file):
    basis_function = Polynomial(degree=1)
    n_regressors = count_model_regressors(
        x=x_train,
        y=y_train,
        xlag = xlag,
        ylag = ylag,
        model_type="NARMAX",
        basis_function=basis_function,
        is_neural_narx=True,
    )
    print(f"n_regressors -> {n_regressors}")

    narx_net = NARXNN(
        net=net(n_regressors),
        xlag = xlag,
        ylag = ylag,
        basis_function=basis_function,
        model_type="NARMAX",
        loss_func="mse_loss",
        optimizer="Adam",
        epochs=2000,
        verbose=False,
        optim_params={
            "betas": (0.9, 0.999),
            "eps": 1e-05,
        }, # optional parameters of the optimizer
    )

    # Save model
    narx_net.fit(X=x_train, y=y_train)
    torch.save(narx_net.net.state_dict(), f"saved_models/narx_net_{y_file}.pth")

    return narx_net

class NARX(nn.Module):
    def __init__(self, n_regressors):
        super().__init__()
        self.lin = nn.Linear(n_regressors, 64)
        self.lin2 = nn.Linear(64, 32)
        self.lin3 = nn.Linear(32, 1)
        self.tanh = nn.Tanh()

    def forward(self, xb):
        z = self.lin(xb)
        z = self.tanh(z)
        z = self.lin2(z)
        z = self.tanh(z)
        z = self.lin3(z)
        return z


if __name__ == "__main__":

    # Parameters
    d = 128
    sample_size = 4096
    lag = 300
    root_dir = "data/raw/cwru"
    files_007_i = ["109", "110", "111", "112"]
    files_007_b = ["122", "123", "124", "125"]
    files_021_i = ["213", "214", "215", "217"]
    files_021_b = ["226", "227", "228", "229"]
    
    x_filenames = ["213", "214", "215", "217"]
    y_filenames = ["109", "110", "111", "112"]

    # Load dataset
    dataset = CWRU()
    for x_filenames, y_filenames in zip([files_021_i, files_021_b, files_007_i, files_007_b], [files_007_i, files_007_b, files_021_i, files_021_b]):
        for x_file, y_file in zip(x_filenames, y_filenames):
            
            print(f"x_file = {x_file}, y_file = {y_file}")

            x_set = dataset.load_file(f"{root_dir}/{x_file}.mat")[0]
            y_set = dataset.load_file(f"{root_dir}/{y_file}.mat")[0]

            x_train = x_set[:sample_size].reshape(-1, 1)
            y_train = y_set[:sample_size].reshape(-1, 1)

            x_valid = x_set[d:(sample_size+d)].reshape(-1, 1)
            y_valid = y_set[d:(sample_size+d)].reshape(-1, 1)

            print(f"x_train.shape = {x_train.shape}, x_valid = {x_valid.shape}, y_train.shape = {y_train.shape}")
                    
            max_size_enable = y_set.shape[0]
            print(f"Max size of subsets: {max_size_enable}")

            # Training
            print(f"Starting the train...")
            net = train(NARX, x_train, y_train, xlag=300, ylag=300, y_file=y_file)

            print(f"Genaration artificial data from file base {y_file}...")
            data_dir = "data/processed/cwru_aug/007/I"
            for i in range(max_size_enable// sample_size):
                yval = y_set[i*sample_size:(i+1)*sample_size].reshape(-1, 1)
                yhat = net.predict(X=x_valid, y=yval)
                np.save(f"{data_dir}/sample_{y_file}_{i}.npy", yhat)
