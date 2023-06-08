import torch
import torch.nn as nn
import numpy as np


# Gaussian Process for Regression
class GaussianProcessRegression(nn.Module):
    def __init__(self, x_dim=1, y_dim=1):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.log_lengthscale = nn.Parameter(torch.tensor(0.0))
        self.log_signal_variance = nn.Parameter(torch.tensor(0.0))
        self.log_observation_noise = nn.Parameter(torch.tensor(0.0))
        
    @property
    def lengthscale(self):
        return torch.exp(self.log_lengthscale)
    
    @property
    def signal_variance(self):
        return torch.exp(self.log_signal_variance)
    
    @property
    def observation_noise(self):
        return torch.exp(self.log_observation_noise)

    def kernel(self, x1, x2):
        """RBFSquaredExponential kernel
        k(x1, x2) = sigma^2 * exp(-0.5 * (x1 - x2)^2 / l^2)
        """
        diff = x1 - x2
        return self.signal_variance * torch.exp(-0.5 * diff ** 2 / self.lengthscale ** 2)

    def fit(self, x, y, n_iter=10000, lr=0.05):
        """Fit the GP to the data"""
        self.x = x
        self.y = y

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-1)
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
        """Negative log-likelihood
        max_{theta} log p(y | x, theta)
        where
        log p(y | x, theta) = -0.5 * logdet(K + sigma^2 I) - 0.5 * y^T K^{-1} y - 0.5 * n * log(2 * pi)
        """

        first_term = 0.5 * torch.logdet(self.K + self.observation_noise * torch.eye(self.K.shape[0]))
        second_term = 0.5 * self.y @ torch.inverse(self.K + self.observation_noise * torch.eye(self.K.shape[0])) @ self.y
        third_term = 0.5 * self.y.numel() * torch.log(2 * torch.tensor(np.pi))
        return first_term + second_term + third_term

    def predict(self, x_star):
        """Predict the mean and variance at x_star
        mean(y_star) = k(x_star, x) K(x, x)^{-1} y
        """
        k_star = self.kernel(x_star.unsqueeze(-1), self.x.unsqueeze(-2))
        k_star_star = self.kernel(x_star.unsqueeze(-1), x_star.unsqueeze(-2))

        mean = k_star @ torch.inverse(self.K + self.observation_noise * torch.eye(self.K.shape[0])) @ self.y
        var = k_star_star - k_star @ torch.inverse(self.K + self.observation_noise * torch.eye(self.K.shape[0])) @ k_star.T
        return mean, var
