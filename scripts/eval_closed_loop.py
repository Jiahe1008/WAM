import argparse
import csv
import os
import sys

import mujoco
import numpy as np
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)

from env.push_env import PushEnv
from model.VLA import VLA
from model.WAM import WAM

HARD_TARGET_RADIUS = 0.10
HARD_TARGET_LOW = (0.55, -0.95)
HARD_TARGET_HIGH = (1.05, 0.95)
HARD_OBJECT_LOW = (-0.85, -0.85)
HARD_OBJECT_HIGH = (0.10, 0.85)
HARD_MIN_OBJECT_TARGET_DIST = 0.75
HARD_PUSHER_BACKOFF = 0.32


def load_checkpoint(path, device):
    """兼容不同 PyTorch 版本加载本地 checkpoint。"""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def split_by_episode(data, train_ratio=0.8, seed=0):
    rng = np.random.default_rng(seed)
    episode_ids = data["episode_ids"]
    unique_episodes = np.unique(episode_ids)

    rng.shuffle(unique_episodes)
    train_num = int(len(unique_episodes) * train_ratio)

    train_episodes = unique_episodes[:train_num]
    val_episodes = unique_episodes[train_num:]

    train_mask = np.isin(episode_ids, train_episodes)
    val_mask = np.isin(episode_ids, val_episodes)

    return train_mask, val_mask


def apply_physics(env, mass, friction):
    env.current_mass = float(mass)
    env.current_friction = float(friction)

    object_geom_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom"
    )
    if object_geom_id >= 0:
        env.model.geom_friction[object_geom_id, 0] = env.current_friction

    if env.object_body_id >= 0:
        env.model.body_mass[env.object_body_id] = env.current_mass

    mujoco.mj_forward(env.model, env.data)


def set_env_state(env, state):
    state = np.asarray(state, dtype=np.float32)
    mujoco.mj_resetData(env.model, env.data)
    env.step_count = 0
    env.target_xy = state[6:8].copy()

    apply_physics(env, state[8], state[9])

    qpos_adr = env.object_qpos_adr
    env.data.qpos[qpos_adr + 0] = state[2]
    env.data.qpos[qpos_adr + 1] = state[3]
    env.data.qpos[qpos_adr + 2] = 0.05
    env.data.qpos[qpos_adr + 3] = 1.0
    env.data.qpos[qpos_adr + 4] = 0.0
    env.data.qpos[qpos_adr + 5] = 0.0
    env.data.qpos[qpos_adr + 6] = 0.0

    qvel_adr = env.object_qvel_adr
    env.data.qvel[qvel_adr:qvel_adr + 6] = 0.0
    env.data.qvel[qvel_adr + 0] = state[4]
    env.data.qvel[qvel_adr + 1] = state[5]

    env.data.mocap_pos[env.pusher_mocap_id][0] = state[0]
    env.data.mocap_pos[env.pusher_mocap_id][1] = state[1]
    env.data.mocap_pos[env.pusher_mocap_id][2] = 0.07

    mujoco.mj_forward(env.model, env.data)


def reset_with_physics(env, mass, friction):
    env.reset()
    apply_physics(env, mass, friction)
    return env.get_state()


def parse_physics_cases(text):
    cases = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue

        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(
                "physics case 格式应为 name:mass:friction，多个 case 用逗号分隔。"
            )

        name, mass, friction = parts
        cases.append(
            {
                "name": name,
                "mass": float(mass),
                "friction": float(friction),
            }
        )

    if not cases:
        raise ValueError("至少需要一个 physics case。")

    return cases


def build_vla_policy(model, state_mean, state_std, device):
    state_mean = np.asarray(state_mean, dtype=np.float32)
    state_std = np.asarray(state_std, dtype=np.float32)

    def policy(state):
        norm_state = (np.asarray(state, dtype=np.float32) - state_mean) / state_std
        state_tensor = torch.tensor(
            norm_state, dtype=torch.float32, device=device
        ).unsqueeze(0)
        with torch.no_grad():
            action = model(state_tensor).squeeze(0).cpu().numpy()
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    return policy


def build_wam_policy(model, state_mean, state_std, device):
    state_mean = np.asarray(state_mean, dtype=np.float32)
    state_std = np.asarray(state_std, dtype=np.float32)

    def policy(state):
        norm_state = (np.asarray(state, dtype=np.float32) - state_mean) / state_std
        state_tensor = torch.tensor(
            norm_state, dtype=torch.float32, device=device
        ).unsqueeze(0)
        dummy_action = torch.zeros((1, 2), dtype=torch.float32, device=device)
        with torch.no_grad():
            action, _, _ = model(state_tensor, dummy_action)
            action = action.squeeze(0).cpu().numpy()
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    return policy


def evaluate_action_mse(model, model_type, checkpoint, data, device, max_samples):
    _, val_mask = split_by_episode(data)
    states = data["states"][val_mask].astype(np.float32)
    actions = data["actions"][val_mask].astype(np.float32)

    if max_samples is not None and len(states) > max_samples:
        states = states[:max_samples]
        actions = actions[:max_samples]

    state_mean = checkpoint["state_mean"].astype(np.float32)
    state_std = checkpoint["state_std"].astype(np.float32)
    norm_states = (states - state_mean) / state_std

    batch_size = 1024
    squared_error_sum = 0.0
    action_count = 0

    model.eval()
    with torch.no_grad():
        for start in range(0, len(norm_states), batch_size):
            end = start + batch_size
            state_tensor = torch.tensor(
                norm_states[start:end], dtype=torch.float32, device=device
            )
            action_tensor = torch.tensor(
                actions[start:end], dtype=torch.float32, device=device
            )

            if model_type == "wam":
                dummy_action = torch.zeros(
                    (len(state_tensor), 2), dtype=torch.float32, device=device
                )
                pred_action, _, _ = model(state_tensor, dummy_action)
            else:
                pred_action = model(state_tensor)

            squared_error_sum += torch.sum((pred_action - action_tensor) ** 2).item()
            action_count += action_tensor.numel()

    return squared_error_sum / action_count


def run_closed_loop(env, initial_state, policy):
    set_env_state(env, initial_state)
    state = env.get_state()
    total_reward = 0.0
    min_distance = env.distance_to_target()
    info = {
        "success": env.is_success(),
        "distance": min_distance,
    }

    for _ in range(env.max_steps):
        action = policy(state)
        state, reward, done, info = env.step(action)
        total_reward += reward
        min_distance = min(min_distance, info["distance"])
        if done:
            break

    return {
        "success": bool(info["success"]),
        "steps": int(env.step_count),
        "final_distance": float(info["distance"]),
        "min_distance": float(min_distance),
        "total_reward": float(total_reward),
    }


def summarize(rows):
    successes = np.asarray([row["success"] for row in rows], dtype=np.float32)
    steps = np.asarray([row["steps"] for row in rows], dtype=np.float32)
    final_distances = np.asarray(
        [row["final_distance"] for row in rows], dtype=np.float32
    )
    min_distances = np.asarray([row["min_distance"] for row in rows], dtype=np.float32)
    rewards = np.asarray([row["total_reward"] for row in rows], dtype=np.float32)

    return {
        "episodes": len(rows),
        "success_rate": float(successes.mean()),
        "mean_steps": float(steps.mean()),
        "mean_final_distance": float(final_distances.mean()),
        "mean_min_distance": float(min_distances.mean()),
        "mean_total_reward": float(rewards.mean()),
    }


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate closed-loop WAM vs VLA.")
    parser.add_argument(
        "--wam-checkpoint",
        default=os.path.join(project_root, "checkpoints", "wam_hard_random_physics_force.pt"),
    )
    parser.add_argument(
        "--vla-checkpoint",
        default=os.path.join(project_root, "checkpoints", "vla_hard_random_physics.pt"),
    )
    parser.add_argument(
        "--vla-matched-checkpoint",
        default=os.path.join(project_root, "checkpoints", "vla_matched_hard_random_physics.pt"),
    )
    parser.add_argument(
        "--data",
        default=os.path.join(project_root, "data", "trajectories_hard_random_physics.npz"),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(project_root, "outputs", "closed_loop_hard_random_physics_force"),
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--wam-hidden-dim", type=int, default=32)
    parser.add_argument("--vla-hidden-dim", type=int, default=32)
    parser.add_argument("--vla-matched-hidden-dim", type=int, default=65)
    parser.add_argument("--target-radius", type=float, default=HARD_TARGET_RADIUS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action-eval-samples", type=int, default=20000)
    parser.add_argument(
        "--physics-cases",
        default=(
            "train:0.2:0.6,"
            "heavy:0.4:0.6,"
            "low_friction:0.2:0.25,"
            "heavy_low_friction:0.4:0.25"
        ),
        help="格式：name:mass:friction，多个 case 用逗号分隔。",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(args.data)

    wam_checkpoint = load_checkpoint(args.wam_checkpoint, device)
    vla_checkpoint = load_checkpoint(args.vla_checkpoint, device)

    wam = WAM(hidden_dim=args.wam_hidden_dim, state_dim=10, action_dim=2).to(device)
    vla = VLA(hidden_dim=args.vla_hidden_dim, state_dim=10, action_dim=2).to(device)
    wam.load_state_dict(wam_checkpoint["model_state_dict"], strict=False)
    vla.load_state_dict(vla_checkpoint["model_state_dict"])
    wam.eval()
    vla.eval()

    policies = {
        "WAM": build_wam_policy(
            wam, wam_checkpoint["state_mean"], wam_checkpoint["state_std"], device
        ),
        "VLA": build_vla_policy(
            vla, vla_checkpoint["state_mean"], vla_checkpoint["state_std"], device
        ),
    }

    action_mse = {
        "WAM": evaluate_action_mse(
            wam, "wam", wam_checkpoint, data, device, args.action_eval_samples
        ),
        "VLA": evaluate_action_mse(
            vla, "vla", vla_checkpoint, data, device, args.action_eval_samples
        ),
    }

    if os.path.exists(args.vla_matched_checkpoint):
        vla_matched_checkpoint = load_checkpoint(args.vla_matched_checkpoint, device)
        vla_matched = VLA(
            hidden_dim=args.vla_matched_hidden_dim,
            state_dim=10,
            action_dim=2,
        ).to(device)
        vla_matched.load_state_dict(vla_matched_checkpoint["model_state_dict"])
        vla_matched.eval()
        policies["VLA-matched"] = build_vla_policy(
            vla_matched,
            vla_matched_checkpoint["state_mean"],
            vla_matched_checkpoint["state_std"],
            device,
        )
        action_mse["VLA-matched"] = evaluate_action_mse(
            vla_matched,
            "vla",
            vla_matched_checkpoint,
            data,
            device,
            args.action_eval_samples,
        )
    else:
        print(f"skip VLA-matched: checkpoint not found: {args.vla_matched_checkpoint}")

    physics_cases = parse_physics_cases(args.physics_cases)
    state_physics_std = data["states"][:, 8:10].std(axis=0)
    has_changed_physics = any(
        abs(case["mass"] - 0.2) > 1e-6 or abs(case["friction"] - 0.6) > 1e-6
        for case in physics_cases
    )
    all_episode_rows = []
    summary_rows = []

    print(f"device: {device}")
    print(f"episodes per case: {args.episodes}")
    print(f"max steps: {args.max_steps}")
    for policy_name, mse in action_mse.items():
        print(f"{policy_name} action MSE: {mse:.6f}")
    if has_changed_physics and np.any(state_physics_std < 1e-5):
        print(
            "warning: training data mass/friction are nearly constant; "
            "changed physics cases are out-of-distribution tests."
        )
    print("-" * 80)

    for case in physics_cases:
        initial_env = PushEnv(
            max_step=args.max_steps,
            seed=args.seed,
            target_radius=args.target_radius,
            target_low=HARD_TARGET_LOW,
            target_high=HARD_TARGET_HIGH,
            object_low=HARD_OBJECT_LOW,
            object_high=HARD_OBJECT_HIGH,
            min_object_target_dist=HARD_MIN_OBJECT_TARGET_DIST,
            pusher_backoff=HARD_PUSHER_BACKOFF,
        )
        initial_states = [
            reset_with_physics(initial_env, case["mass"], case["friction"])
            for _ in range(args.episodes)
        ]

        for policy_name, policy in policies.items():
            env = PushEnv(
                max_step=args.max_steps,
                seed=args.seed,
                target_radius=args.target_radius,
                target_low=HARD_TARGET_LOW,
                target_high=HARD_TARGET_HIGH,
                object_low=HARD_OBJECT_LOW,
                object_high=HARD_OBJECT_HIGH,
                min_object_target_dist=HARD_MIN_OBJECT_TARGET_DIST,
                pusher_backoff=HARD_PUSHER_BACKOFF,
            )
            case_rows = []

            for episode_id, initial_state in enumerate(initial_states):
                result = run_closed_loop(env, initial_state, policy)
                row = {
                    "physics_case": case["name"],
                    "policy": policy_name,
                    "episode_id": episode_id,
                    "mass": case["mass"],
                    "friction": case["friction"],
                    **result,
                }
                case_rows.append(row)
                all_episode_rows.append(row)

            summary = summarize(case_rows)
            summary_row = {
                "physics_case": case["name"],
                "policy": policy_name,
                "mass": case["mass"],
                "friction": case["friction"],
                "action_mse": action_mse[policy_name],
                **summary,
            }
            summary_rows.append(summary_row)

            print(
                f"{case['name']} | {policy_name} | "
                f"success={summary['success_rate']:.3f} | "
                f"steps={summary['mean_steps']:.2f} | "
                f"final_dist={summary['mean_final_distance']:.4f} | "
                f"min_dist={summary['mean_min_distance']:.4f}"
            )
        print("-" * 80)

    episode_path = os.path.join(args.output_dir, "closed_loop_episodes.csv")
    summary_path = os.path.join(args.output_dir, "closed_loop_summary.csv")
    for path in (episode_path, summary_path):
        if os.path.exists(path):
            raise FileExistsError(f"{path} already exists; refusing to overwrite it.")

    write_csv(
        episode_path,
        all_episode_rows,
        [
            "physics_case",
            "policy",
            "episode_id",
            "mass",
            "friction",
            "success",
            "steps",
            "final_distance",
            "min_distance",
            "total_reward",
        ],
    )
    write_csv(
        summary_path,
        summary_rows,
        [
            "physics_case",
            "policy",
            "mass",
            "friction",
            "action_mse",
            "episodes",
            "success_rate",
            "mean_steps",
            "mean_final_distance",
            "mean_min_distance",
            "mean_total_reward",
        ],
    )

    print("saved episode results to:", episode_path)
    print("saved summary to:", summary_path)


if __name__ == "__main__":
    main()
