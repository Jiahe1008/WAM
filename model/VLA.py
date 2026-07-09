import torch
import torch.nn as nn

class VLA(nn.Module):
    def __init__(self, hidden_dim=64, state_dim=10, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
    def forward(self, state):
        return self.net(state)