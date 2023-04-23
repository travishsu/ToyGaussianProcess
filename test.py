import numpy as np
import torch
import matplotlib.pyplot as plt
from model import GaussianProcessRegression

import argparse


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument("--n_samples", type=int, default=100)
    args.add_argument("--xmin", type=float, default=0)
    args.add_argument("--xmax", type=float, default=1)
    args.add_argument("--n_iter", type=int, default=10000)
    args.add_argument("--lr", type=float, default=0.05)

    return args.parse_args()


def get_data(n_samples=100, xmin=0, xmax=1, seed=0, noise=0.9):
    """ y = sin(2*pi*x) + epsilon, epsilon ~ N(0, noise)

    Args:
        n_samples (int, optional): _description_. Defaults to 100.
    """
    x = np.linspace(xmin, xmax, n_samples)
    y = np.sin(2 * np.pi * x) + np.random.normal(0, noise, size=n_samples)
    return x, y


if __name__ == "__main__":
    args = parse_args()

    train_x, train_y = get_data(n_samples=args.n_samples, xmin=args.xmin, xmax=args.xmax)
    train_x = torch.tensor(train_x, dtype=torch.float32)
    train_y = torch.tensor(train_y, dtype=torch.float32)

    gp = GaussianProcessRegression(x_dim=1, y_dim=1)
    gp.fit(train_x, train_y, n_iter=args.n_iter, lr=args.lr)

    # Plot the data and the predictions
    testx = torch.linspace(0, 1, 100)
    pred, var = gp.predict(testx)

    plt.plot(train_x, train_y, "o")
    plt.plot(testx.numpy(), pred.detach().numpy())

    # Plot the confidence interval
    var = var.diag()
    plt.fill_between(testx.numpy(), (pred - 10 * var).detach().numpy(), (pred + 10 * var).detach().numpy(), alpha=0.2)

    plt.show()
