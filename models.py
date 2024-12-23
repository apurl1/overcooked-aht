import torch
import torch.nn as nn
from torch import Tensor


class NNet(nn.Module):
    def __init__(self, input_size: int, action_dim: int, feature_dim: int):
        """creates a neural network with linear layers and ReLU activation

        Args:
            input_size (int): size of state observation from env
            action_dim (int): size of env action space
            feature_dim (int): number of features
        """
        super(NNet, self).__init__()
        self.input_size = input_size
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        layers = []
        layers.append(nn.Linear(input_size, 64, device=self.device))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(64, 128, device=self.device))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(128, action_dim * feature_dim, device=self.device))

        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """predict action values given state input

        Args:
            x (Tensor): state obs from env

        Returns:
            Tensor: predicted values with shape (128, action_dim, features_dim)
        """
        x = x.view(-1, self.input_size).float()
        #print(x.shape)
        output: Tensor = self.model(x)
        return output.view([output.shape[0], self.action_dim, self.feature_dim])
