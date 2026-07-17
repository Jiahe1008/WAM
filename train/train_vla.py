import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)

from data.dataset import VLADataset
from model.VLA import VLA

DATA_FILENAME = "trajectories_hard_random_physics.npz"
CHECKPOINT_FILENAME = "vla_hard_random_physics.pt"
HISTORY_FILENAME = "vla_hard_random_physics_history.json"
MATCHED_HIDDEN_DIM = 65
MATCHED_CHECKPOINT_FILENAME = "vla_matched_hard_random_physics.pt"
MATCHED_HISTORY_FILENAME = "vla_matched_hard_random_physics_history.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Train VLA baseline.")
    parser.add_argument("--data-filename", default=DATA_FILENAME)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--checkpoint-filename", default=CHECKPOINT_FILENAME)
    parser.add_argument("--history-filename", default=HISTORY_FILENAME)
    parser.add_argument(
        "--matched",
        action="store_true",
        help="Train a parameter-matched VLA baseline for force-head WAM hidden_dim=32.",
    )
    args = parser.parse_args()

    if args.matched:
        args.hidden_dim = MATCHED_HIDDEN_DIM
        args.checkpoint_filename = MATCHED_CHECKPOINT_FILENAME
        args.history_filename = MATCHED_HISTORY_FILENAME

    return args


def save_history_json(path, history, best_epoch, best_val_loss):
    payload = {
        "history": history,
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val_loss),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


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
    args = parse_args()
    data_path = os.path.join(project_root, "data", args.data_filename)
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
    state_std = train_states.std(axis=0) + 1e-6
    
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

    model = VLA(hidden_dim=args.hidden_dim, state_dim=10, action_dim=2).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print("hidden_dim:", args.hidden_dim)
    print("param_count:", param_count)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    num_epochs = args.num_epochs
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
    }
    best_val_loss = float("inf")
    best_epoch = -1
    best_model_state_dict = None

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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(
                f"epoch={epoch:03d} | "
                f"train_loss={train_loss:.6f} | "
                f"val_loss={val_loss:.6f}"
            )

    save_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, args.checkpoint_filename)
    history_path = os.path.join(save_dir, args.history_filename)
    for path in (save_path, history_path):
        if os.path.exists(path):
            raise FileExistsError(f"{path} already exists; refusing to overwrite it.")

    torch.save(
        {
            "model_state_dict": best_model_state_dict,
            "state_mean": state_mean,
            "state_std": state_std,
            "history": history,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
        },
        save_path,
    )
    save_history_json(history_path, history, best_epoch, best_val_loss)

    print("saved model to:", save_path)
    print(f"best epoch: {best_epoch}, best val_loss: {best_val_loss:.6f}")
    print("saved history to:", history_path)


if __name__ == "__main__":
    main()
