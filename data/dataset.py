import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)

class VLADataset(Dataset):
    # dataset子类必须具有__len__, __getitem__ 方法
    """从 trajectories.npz 读取 state-action 数据"""

    def __init__(self, states, actions, state_mean, state_std):
        self.states = states.astype(np.float32)
        self.actions = actions.astype(np.float32)

        self.state_mean = state_mean.astype(np.float32)
        self.state_std = state_std.astype(np.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = self.states[idx]
        action = self.actions[idx]

        # 状态归一化
        state = (state - self.state_mean) / self.state_std

        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(action, dtype=torch.float32),
        )


class WAMDataset(Dataset):
    """WAM 数据集：state, action, next_state"""

    def __init__(self, states, actions, next_states, state_mean, state_std):
        self.states = states.astype(np.float32)
        self.actions = actions.astype(np.float32)
        self.next_states = next_states.astype(np.float32)

        self.state_mean = state_mean.astype(np.float32)
        self.state_std = state_std.astype(np.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = self.states[idx]
        action = self.actions[idx]
        next_state = self.next_states[idx]

        # state 和 next_state 用同一套归一化参数
        state = (state - self.state_mean) / self.state_std
        next_state = (next_state - self.state_mean) / self.state_std

        return (
            torch.tensor(state, dtype=torch.float32),
            torch.tensor(action, dtype=torch.float32),
            torch.tensor(next_state, dtype=torch.float32),
        )