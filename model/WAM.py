import torch
import torch.nn as nn

class WAM(nn.Module):
    def __init__(self, hidden_dim=64, state_dim=10, action_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),

        )
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        self.state_head = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
    def forward(self, state, action):
        hidden_state = self.net(state)
        pred_action = self.action_head(hidden_state)
        
        # hidden_state [B, hidden_dim,]
        # action [B, 2,]
        world_input = torch.cat([hidden_state, action], dim=1)
        delta_pred_state = self.state_head(world_input) 
        pred_next_state = state + delta_pred_state
        return pred_action, pred_next_state