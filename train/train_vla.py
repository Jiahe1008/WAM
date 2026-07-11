import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)

from data.dataset import VLADataset
from model.VLA import VLA

def split_by_episode(data, train_ratio=0.8, seed=0):
    rng = np.random.default_rng(seed)
    episode_ids = data["episode_ids"]  
    unique_episodes = np.unique(episode_ids)  # 找出共有多少个episode (N,)

    rng.shuffle(unique_episodes)
    train_num = int(len(unique_episodes) * train_ratio)

    train_episodes = unique_episodes[train_num:]
    val_episodes = unique_episodes[:train_num]

    train_mask = np.isin(episode_ids, train_episodes)
    val_mask = np.isin(episode_ids, val_episodes)
    
    return train_mask, val_mask

def main():
    data_path = os.path.join(project_root, "data", "trajectories.npz")
    data = np.load(data_path)

    state = data["states"]
    action = data["actions"]

    train_mask, val_mask = split_by_episode(data)
    train_states = state[train_mask]
    train_actions = action[train_mask]
    val_states = state[val_mask]
    val_actions = action[val_mask]

    print("train states:", train_states.shape)
    print("val states:", val_states.shape)
    
    state_mean = train_states.mean(axis=0)
    state_std = train_states.std(axis=0) + 1e-6\
    
    train_dataset = VLADataset(train_states, train_actions, state_mean, state_std)
    val_dataset = VLADataset(val_states, val_actions, state_mean, state_std)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    model = VLA(hidden_dim=32, state_dim=10, action_dim=2).to(device)

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
        train_loss_sum = 0
        train_count = 0

        for batch_states, batch_action in train_loader:
            batch_states = batch_states.to(device)
            batch_actions = batch_action.to(device)

            pred_actions = model(batch_states)
            loss = criterion(pred_actions, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(batch_states)
            train_count += len(batch_states)
        
        train_loss = train_loss_sum / train_count

        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        
        with torch.no_grad():
            for batch_states, batch_actions in val_loader:
                batch_states = batch_states.to(device)
                batch_actions = batch_actions.to(device)

                pred_actions = model(batch_states)
                loss = criterion(pred_actions, batch_actions)

                val_loss_sum += loss.item() * len(batch_states)
                val_count += len(batch_states)

        val_loss = val_loss_sum / val_count

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(
                f"epoch={epoch:03d} | "
                f"train_loss={train_loss:.6f} | "
                f"val_loss={val_loss:.6f}"
            )

    save_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "vla_baseline.pt")
    history_path = os.path.join(save_dir, "vla_baseline_history.npz")

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
