import numpy as np
import torch
from torch import nn


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def mlp(input_dim, output_dim, hidden_dim=64):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class AssociationActor(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, observations):
        hidden = self.backbone(observations)
        return self.mean(hidden), self.log_std(hidden).clamp(
            LOG_STD_MIN, LOG_STD_MAX
        )

    def sample(self, observations, deterministic=False):
        mean, log_std = self(observations)
        if deterministic:
            actions = torch.tanh(mean)
            return actions, None
        distribution = torch.distributions.Normal(mean, log_std.exp())
        raw_actions = distribution.rsample()
        actions = torch.tanh(raw_actions)
        log_probability = distribution.log_prob(raw_actions)
        log_probability -= torch.log(1 - actions.square() + 1e-6)
        return actions, log_probability.sum(dim=-1, keepdim=True)


class TwinAssociationCritic(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_dim=64):
        super().__init__()
        input_dim = observation_dim + action_dim
        self.q1 = mlp(input_dim, 1, hidden_dim)
        self.q2 = mlp(input_dim, 1, hidden_dim)

    def forward(self, observations, actions):
        inputs = torch.cat((observations, actions), dim=-1)
        return self.q1(inputs), self.q2(inputs)


def initialize_linear_layers(module):
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.zeros_(layer.bias)
