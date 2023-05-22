import torch as th
import torch.nn as nn
from gym.spaces import Discrete
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Discrete, features_dim: int=128,
        hidden_size = [16, 32], kernel = 3, stride = 2):

        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)

        # --- n_input_channels for grey scale is 1.
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            nn.BatchNorm2d(n_input_channels),
            nn.Conv2d(n_input_channels, hidden_size[0], kernel, stride),
            nn.BatchNorm2d(hidden_size[0]),
            nn.Tanh(),
            nn.MaxPool2d(kernel, stride),
            nn.Conv2d(hidden_size[0], hidden_size[1], kernel, stride),
            nn.BatchNorm2d(hidden_size[1]),
            nn.Tanh(),
            nn.MaxPool2d(kernel, stride),
            nn.Flatten(),
            # After flatten layer, vector has length 128.
        )     

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.cnn(observations)