import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_actions):
        super(DQN, self).__init__()

        self.liner_layer1 = nn.Linear(state_dim, hidden_dim)

        self.liner_layer2 = nn.Linear(hidden_dim, hidden_dim)

        self.output_layer = nn.Linear(hidden_dim, num_actions)

    def forward(self, x):
        x = torch.relu(self.liner_layer1(x))
        x = torch.relu(self.liner_layer2(x))
        x = self.output_layer(x)
