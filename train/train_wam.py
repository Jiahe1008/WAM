import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)

from data.dataset import WAMDataset
from model.WAM import WAM

lambda_pred = 1.0
lambda_phys = 0.01

def compute_physics_loss(norm_states, norm_pred_next_states, state_mean, state_std):
    """
    norm_states:           [B, 10] 归一化后的当前状态
    norm_pred_next_states: [B, 10] 归一化后的预测下一状态
    state_mean/state_std:  [10]
    """

    # 反归一化，物理约束最好在真实物理量上算
    states = norm_states * state_std + state_mean
    pred_next_states = norm_pred_next_states * state_std + state_mean

    # 当前状态
    cur_v = states[:, 4:6]

    # 预测下一状态
    pred_pusher_xy = pred_next_states[:, 0:2]
    pred_object_xy = pred_next_states[:, 2:4]
    pred_v = pred_next_states[:, 4:6]

    # -----------------------------
    # 1. 穿透惩罚
    # -----------------------------
    pusher_radius = 0.06
    object_radius = 0.08
    min_dist = pusher_radius + object_radius

    dist = torch.norm(pred_pusher_xy - pred_object_xy, dim=-1)

    penetration = torch.relu(min_dist - dist)

    loss_penetration = (penetration ** 2).mean()

    # -----------------------------
    # 2. 速度平滑约束
    # -----------------------------
    velocity_change = pred_v - cur_v

    loss_smooth = (velocity_change ** 2).mean()

    # 权重可以在这里内部调，也可以外部调
    loss_phys = loss_penetration + 0.1 * loss_smooth

    return loss_phys


def split_by_episode(data, train_ratio=0.8, seed=0):
    rng = np.random.default_rng(seed)
    episode_ids = data["episode_ids"]  
    unique_episodes = np.unique(episode_ids)  # 找出共有多少个episode (N,)

    rng.shuffle(unique_episodes)
    train_num = int(len(unique_episodes) * train_ratio)

    train_episodes = unique_episodes[:train_num]
    val_episodes = unique_episodes[train_num:]

    train_mask = np.isin(episode_ids, train_episodes)
    val_mask = np.isin(episode_ids, val_episodes)
    
    return train_mask, val_mask

def main():
    data_path = os.path.join(project_root, "data", "trajectories.npz")
    data = np.load(data_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    state = data["states"]
    action = data["actions"]
    next_states = data["next_states"]

    train_mask, val_mask = split_by_episode(data)
    train_states = state[train_mask]
    train_actions = action[train_mask]
    val_states = state[val_mask]
    val_actions = action[val_mask]
    train_next_states = next_states[train_mask]
    val_next_states = next_states[val_mask]

    print("train states:", train_states.shape)
    print("val states:", val_states.shape)
    
    state_mean = train_states.mean(axis=0)
    state_std = train_states.std(axis=0) + 1e-6
    state_mean_tensor = torch.tensor(state_mean, dtype=torch.float32, device=device)
    state_std_tensor = torch.tensor(state_std, dtype=torch.float32, device=device)
    train_dataset = WAMDataset(train_states, train_actions, train_next_states, state_mean, state_std)
    val_dataset = WAMDataset(val_states, val_actions, val_next_states, state_mean, state_std)

    train_loader = DataLoader(
        train_dataset,
        batch_size= 128,
        shuffle=True,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=True,
        drop_last=False
    )



    model = WAM(hidden_dim=32, state_dim=10, action_dim=2).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    num_epochs = 50
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
    }
    for epoch in range(num_epochs):
        model.train()

        train_loss_sum = 0.0
        train_action_loss_sum = 0.0
        train_pred_loss_sum = 0.0
        train_phys_loss_sum = 0.0
        train_count = 0

        for batch_states, batch_actions, batch_next_states in train_loader:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)
            batch_next_states = batch_next_states.to(device)

            pred_actions, pred_next_states = model(batch_states, batch_actions)

            loss_action = criterion(pred_actions, batch_actions)
            loss_pred = criterion(pred_next_states, batch_next_states)

            loss_phys = compute_physics_loss(
                batch_states,
                pred_next_states,
                state_mean_tensor,
                state_std_tensor,
            )

            loss = loss_action + lambda_pred * loss_pred + lambda_phys * loss_phys

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = len(batch_states)

            train_loss_sum += loss.item() * batch_size
            train_action_loss_sum += loss_action.item() * batch_size
            train_pred_loss_sum += loss_pred.item() * batch_size
            train_phys_loss_sum += loss_phys.item() * batch_size
            train_count += batch_size

        train_loss = train_loss_sum / train_count
        train_action_loss = train_action_loss_sum / train_count
        train_pred_loss = train_pred_loss_sum / train_count
        train_phys_loss = train_phys_loss_sum / train_count

        model.eval()

        val_loss_sum = 0.0
        val_action_loss_sum = 0.0
        val_pred_loss_sum = 0.0
        val_phys_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for batch_states, batch_actions, batch_next_states in val_loader:
                batch_states = batch_states.to(device)
                batch_actions = batch_actions.to(device)
                batch_next_states = batch_next_states.to(device)

                pred_actions, pred_next_states = model(batch_states, batch_actions)

                loss_action = criterion(pred_actions, batch_actions)
                loss_pred = criterion(pred_next_states, batch_next_states)

                loss_phys = compute_physics_loss(
                    batch_states,
                    pred_next_states,
                    state_mean_tensor,
                    state_std_tensor,
                )

                loss = loss_action + lambda_pred * loss_pred + lambda_phys * loss_phys

                batch_size = len(batch_states)

                val_loss_sum += loss.item() * batch_size
                val_action_loss_sum += loss_action.item() * batch_size
                val_pred_loss_sum += loss_pred.item() * batch_size
                val_phys_loss_sum += loss_phys.item() * batch_size
                val_count += batch_size

        val_loss = val_loss_sum / val_count
        val_action_loss = val_action_loss_sum / val_count
        val_pred_loss = val_pred_loss_sum / val_count
        val_phys_loss = val_phys_loss_sum / val_count
    
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(
                f"epoch={epoch:03d} | "
                f"train_loss={train_loss:.6f} | "
                f"train_action={train_action_loss:.6f} | "
                f"train_pred={train_pred_loss:.6f} | "
                f"train_phys={train_phys_loss:.6f} | "
                f"val_loss={val_loss:.6f} | "
                f"val_action={val_action_loss:.6f} | "
                f"val_pred={val_pred_loss:.6f} | "
                f"val_phys={val_phys_loss:.6f}"
            )

    save_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "wam.pt")
    history_path = os.path.join(save_dir, "wam_history.npz")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "state_mean": state_mean,
            "state_std": state_std,
            "history": history,
        },
        save_path,
    )
    np.savez(history_path, **history)

    print("saved model to:", save_path)
    print("saved history to:", history_path)


if __name__ == "__main__":
    main()
