import torch
import torch.nn as nn

USE_CUDA = torch.cuda.is_available()


class DQN(nn.Module):
    def __init__(self, state_dim, hidden_dim1, hidden_dim2, num_actions):
        super(DQN, self).__init__()

        self.liner_layer1 = nn.Linear(state_dim, hidden_dim1)

        self.liner_layer2 = nn.Linear(hidden_dim1, hidden_dim2)

        self.output_layer = nn.Linear(hidden_dim2, num_actions)

        if USE_CUDA:
            self.liner_layer1 = self.liner_layer1.cuda()
            self.liner_layer2 = self.liner_layer2.cuda()
            self.output_layer = self.output_layer.cuda()

    def forward(self, x):
        hidden_vector1 = torch.relu(self.liner_layer1(x))
        hidden_vector2 = torch.relu(self.liner_layer2(hidden_vector1))
        action_value = self.output_layer(hidden_vector2)
        return action_value
