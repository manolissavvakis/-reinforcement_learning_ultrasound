import torch as th
import torch.nn as nn
from gym.spaces import Discrete
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self,
                observation_space: Discrete,
                features_dim: int=64,
                hidden_size = [16, 32],
                kernel = 3,
                stride = 2
                ):

        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Grey scale images have 1 channel.
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, hidden_size[0], kernel, stride),
            nn.BatchNorm2d(hidden_size[0]),
            nn.Tanh(), # Check for ReLU
            nn.MaxPool2d(kernel, stride),
            nn.Conv2d(hidden_size[0], hidden_size[1], kernel, stride),
            nn.BatchNorm2d(hidden_size[1]),
            nn.Tanh(), # Check for ReLU
            nn.MaxPool2d(kernel, stride),
            nn.Flatten(),
            # After flatten layer, vector has length 128.
        )
        
        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))