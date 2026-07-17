import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)

from data.dataset import WAMDataset
from model.WAM import WAM
from physical_loss import compute_dynamics_residual_loss, compute_physics_loss

DATA_FILENAME = "trajectories_hard_random_physics.npz"
CHECKPOINT_FILENAME = "wam_hard_random_physics_force.pt"
HISTORY_FILENAME = "wam_hard_random_physics_force_history.json"

lambda_pred = 1.0
lambda_phys = 0.01
lambda_force = 0.001


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
    data_path = os.path.join(project_root, "data", DATA_FILENAME)
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
    param_count = sum(p.numel() for p in model.parameters())
    print("hidden_dim:", 32)
    print("param_count:", param_count)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    num_epochs = 100
    history = {
        "epoch": [],
        "train_loss": [],
        "train_action_loss": [],
        "train_pred_loss": [],
        "train_phys_loss": [],
        "train_force_loss": [],
        "weighted_train_phys_loss": [],
        "weighted_train_force_loss": [],
        "val_loss": [],
        "val_action_loss": [],
        "val_pred_loss": [],
        "val_phys_loss": [],
        "val_force_loss": [],
        "weighted_val_phys_loss": [],
        "weighted_val_force_loss": [],
    }
    best_val_loss = float("inf")
    best_epoch = -1
    best_model_state_dict = None

    for epoch in range(num_epochs):
        model.train()

        train_loss_sum = 0.0
        train_action_loss_sum = 0.0
        train_pred_loss_sum = 0.0
        train_phys_loss_sum = 0.0
        train_force_loss_sum = 0.0
        train_count = 0

        for batch_states, batch_actions, batch_next_states in train_loader:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)
            batch_next_states = batch_next_states.to(device)

            pred_actions, pred_next_states, pred_contact_force = model(
                batch_states,
                batch_actions,
            )

            loss_action = criterion(pred_actions, batch_actions)
            loss_pred = criterion(pred_next_states, batch_next_states)

            loss_phys = compute_physics_loss(
                batch_states,
                pred_next_states,
                state_mean_tensor,
                state_std_tensor,
            )

            loss_force = compute_dynamics_residual_loss(
                batch_states,
                batch_next_states,
                pred_contact_force,
                state_mean_tensor,
                state_std_tensor,
            )

            loss = (
                loss_action
                + lambda_pred * loss_pred
                + lambda_phys * loss_phys
                + lambda_force * loss_force
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = len(batch_states)

            train_loss_sum += loss.item() * batch_size
            train_action_loss_sum += loss_action.item() * batch_size
            train_pred_loss_sum += loss_pred.item() * batch_size
            train_phys_loss_sum += loss_phys.item() * batch_size
            train_force_loss_sum += loss_force.item() * batch_size
            train_count += batch_size

        train_loss = train_loss_sum / train_count
        train_action_loss = train_action_loss_sum / train_count
        train_pred_loss = train_pred_loss_sum / train_count
        train_phys_loss = train_phys_loss_sum / train_count
        train_force_loss = train_force_loss_sum / train_count
        weighted_train_phys_loss = lambda_phys * train_phys_loss
        weighted_train_force_loss = lambda_force * train_force_loss

        model.eval()

        val_loss_sum = 0.0
        val_action_loss_sum = 0.0
        val_pred_loss_sum = 0.0
        val_phys_loss_sum = 0.0
        val_force_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for batch_states, batch_actions, batch_next_states in val_loader:
                batch_states = batch_states.to(device)
                batch_actions = batch_actions.to(device)
                batch_next_states = batch_next_states.to(device)

                pred_actions, pred_next_states, pred_contact_force = model(
                    batch_states,
                    batch_actions,
                )

                loss_action = criterion(pred_actions, batch_actions)
                loss_pred = criterion(pred_next_states, batch_next_states)

                loss_phys = compute_physics_loss(
                    batch_states,
                    pred_next_states,
                    state_mean_tensor,
                    state_std_tensor,
                )

                loss_force = compute_dynamics_residual_loss(
                    batch_states,
                    batch_next_states,
                    pred_contact_force,
                    state_mean_tensor,
                    state_std_tensor,
                )

                loss = (
                    loss_action
                    + lambda_pred * loss_pred
                    + lambda_phys * loss_phys
                    + lambda_force * loss_force
                )

                batch_size = len(batch_states)

                val_loss_sum += loss.item() * batch_size
                val_action_loss_sum += loss_action.item() * batch_size
                val_pred_loss_sum += loss_pred.item() * batch_size
                val_phys_loss_sum += loss_phys.item() * batch_size
                val_force_loss_sum += loss_force.item() * batch_size
                val_count += batch_size

        val_loss = val_loss_sum / val_count
        val_action_loss = val_action_loss_sum / val_count
        val_pred_loss = val_pred_loss_sum / val_count
        val_phys_loss = val_phys_loss_sum / val_count
        val_force_loss = val_force_loss_sum / val_count
        weighted_val_phys_loss = lambda_phys * val_phys_loss
        weighted_val_force_loss = lambda_force * val_force_loss

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_action_loss"].append(train_action_loss)
        history["train_pred_loss"].append(train_pred_loss)
        history["train_phys_loss"].append(train_phys_loss)
        history["train_force_loss"].append(train_force_loss)
        history["weighted_train_phys_loss"].append(weighted_train_phys_loss)
        history["weighted_train_force_loss"].append(weighted_train_force_loss)
        history["val_loss"].append(val_loss)
        history["val_action_loss"].append(val_action_loss)
        history["val_pred_loss"].append(val_pred_loss)
        history["val_phys_loss"].append(val_phys_loss)
        history["val_force_loss"].append(val_force_loss)
        history["weighted_val_phys_loss"].append(weighted_val_phys_loss)
        history["weighted_val_force_loss"].append(weighted_val_force_loss)

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
                f"train_action={train_action_loss:.6f} | "
                f"train_pred={train_pred_loss:.6f} | "
                f"train_phys={train_phys_loss:.6f} | "
                f"train_force={train_force_loss:.6f} | "
                f"w_train_phys={weighted_train_phys_loss:.6f} | "
                f"w_train_force={weighted_train_force_loss:.6f} | "
                f"val_loss={val_loss:.6f} | "
                f"val_action={val_action_loss:.6f} | "
                f"val_pred={val_pred_loss:.6f} | "
                f"val_phys={val_phys_loss:.6f} | "
                f"val_force={val_force_loss:.6f} | "
                f"w_val_phys={weighted_val_phys_loss:.6f} | "
                f"w_val_force={weighted_val_force_loss:.6f}"
            )

    save_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, CHECKPOINT_FILENAME)
    history_path = os.path.join(save_dir, HISTORY_FILENAME)
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
