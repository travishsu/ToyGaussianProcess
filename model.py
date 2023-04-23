import torch
import torch.nn as nn
import numpy as np


# Gaussian Process for Regression
class GaussianProcessRegression(nn.Module):
    def __init__(self, x_dim=1, y_dim=1):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.lengthscale = nn.Parameter(torch.tensor(1.0))
        self.signal_variance = nn.Parameter(torch.tensor(1.0))
        self.observation_noise = nn.Parameter(torch.tensor(1.0))

    def kernel(self, x1, x2):
        """RBFSquaredExponential kernel"""
        diff = x1 - x2
        return self.signal_variance * torch.exp(-0.5 * diff ** 2 / self.lengthscale ** 2)

    def fit(self, x, y, n_iter=10000, lr=0.05):
        """Fit the GP to the data"""
        self.x = x
        self.y = y

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        for iter in range(n_iter):
            optimizer.zero_grad()
            # Compute the kernel matrix
            self.K = self.kernel(x.unsqueeze(-1), x.unsqueeze(-2))
            loss = self.loss_func()
            loss.backward()
            optimizer.step()

            # Print the loss every 100 iterations
            if iter % 100 == 0:
                print(f"iter {iter}: loss = {loss.item()}, lengthscale = {self.lengthscale.item()}, signal_variance = {self.signal_variance.item()}, observation_noise = {self.observation_noise.item()}")

    def loss_func(self):
        """Negative log-likelihood"""
        first_term = 0.5 * torch.logdet(self.K + self.observation_noise * torch.eye(self.K.shape[0]))
        second_term = 0.5 * self.y @ torch.inverse(self.K + self.observation_noise * torch.eye(self.K.shape[0])) @ self.y
        third_term = 0.5 * self.y.numel() * torch.log(2 * torch.tensor(np.pi))
        return first_term + second_term + third_term

    def predict(self, x_star):
        """Predict the mean and variance at x_star"""
        k_star = self.kernel(x_star.unsqueeze(-1), self.x.unsqueeze(-2))
        k_star_star = self.kernel(x_star.unsqueeze(-1), x_star.unsqueeze(-2))

        mean = k_star @ torch.inverse(self.K + self.observation_noise * torch.eye(self.K.shape[0])) @ self.y
        var = k_star_star - k_star @ torch.inverse(self.K + self.observation_noise * torch.eye(self.K.shape[0])) @ k_star.T
        return mean, var
