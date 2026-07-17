import argparse
import os
import sys

import matplotlib.pyplot as plt
import mujoco
import numpy as np
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)

from env.push_env import PushEnv
from model.WAM import WAM


PUSHER_RADIUS = 0.06
OBJECT_RADIUS = 0.08
MIN_CONTACT_DIST = PUSHER_RADIUS + OBJECT_RADIUS
TARGET_RADIUS = 0.10
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


def set_env_state(env, state):
    """把 PushEnv 恢复到一个低维 state，方便和 WAM 从同一初始条件 rollout。"""
    state = np.asarray(state, dtype=np.float32)
    env.step_count = 0
    env.target_xy = state[6:8].copy()
    env.current_mass = float(state[8])
    env.current_friction = float(state[9])

    object_geom_id = mujoco.mj_name2id(
        env.model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom"
    )
    if object_geom_id >= 0:
        env.model.geom_friction[object_geom_id, 0] = env.current_friction

    if env.object_body_id >= 0:
        env.model.body_mass[env.object_body_id] = env.current_mass

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


def mujoco_rollout(env, initial_state, actions):
    """用 MuJoCo 真实动力学执行动作序列，返回长度为 horizon + 1 的状态序列。"""
    set_env_state(env, initial_state)

    states = [env.get_state()]
    for action in actions:
        next_state, _, done, _ = env.step(action)
        states.append(next_state)
        if done:
            break

    return np.asarray(states, dtype=np.float32)


def wam_rollout(model, initial_state, actions, state_mean, state_std, device):
    """开环预测：只给初始状态，之后每一步都使用 WAM 上一步预测出的状态。"""
    model.eval()

    state_mean = torch.tensor(state_mean, dtype=torch.float32, device=device)
    state_std = torch.tensor(state_std, dtype=torch.float32, device=device)

    real_state = np.asarray(initial_state, dtype=np.float32)
    norm_state = (real_state - state_mean.cpu().numpy()) / state_std.cpu().numpy()
    norm_state = torch.tensor(norm_state, dtype=torch.float32, device=device).unsqueeze(0)

    pred_states = [real_state.copy()]
    with torch.no_grad():
        for action in actions:
            action_tensor = torch.tensor(
                action, dtype=torch.float32, device=device
            ).unsqueeze(0)
            _, norm_next_state, _ = model(norm_state, action_tensor)

            real_next_state = (
                norm_next_state.squeeze(0) * state_std + state_mean
            ).cpu().numpy()

            pred_states.append(real_next_state.astype(np.float32))
            norm_state = norm_next_state

    return np.asarray(pred_states, dtype=np.float32)


def penetration_rate(states):
    pusher_xy = states[:, 0:2]
    object_xy = states[:, 2:4]
    dist = np.linalg.norm(pusher_xy - object_xy, axis=1)
    return float(np.mean(dist < MIN_CONTACT_DIST))


def mean_penetration_depth(states):
    pusher_xy = states[:, 0:2]
    object_xy = states[:, 2:4]
    dist = np.linalg.norm(pusher_xy - object_xy, axis=1)
    depth = np.maximum(MIN_CONTACT_DIST - dist, 0.0)
    return float(np.mean(depth))


def velocity_jump(states):
    """用相邻速度差近似速度连续性误差，越小越平滑。"""
    if len(states) < 2:
        return 0.0
    vel = states[:, 4:6]
    return float(np.linalg.norm(np.diff(vel, axis=0), axis=1).mean())


def compute_metrics(true_states, pred_states, horizons):
    max_len = min(len(true_states), len(pred_states))
    true_states = true_states[:max_len]
    pred_states = pred_states[:max_len]

    metrics = {
        "pred_penetration_rate": penetration_rate(pred_states),
        "pred_penetration_depth": mean_penetration_depth(pred_states),
        "pred_velocity_jump": velocity_jump(pred_states),
        "true_velocity_jump": velocity_jump(true_states),
    }

    for horizon in horizons:
        if horizon >= max_len:
            continue
        object_error = np.linalg.norm(
            pred_states[horizon, 2:4] - true_states[horizon, 2:4]
        )
        pusher_error = np.linalg.norm(
            pred_states[horizon, 0:2] - true_states[horizon, 0:2]
        )
        state_rmse = np.sqrt(np.mean((pred_states[horizon] - true_states[horizon]) ** 2))

        metrics[f"object_error_{horizon}"] = float(object_error)
        metrics[f"pusher_error_{horizon}"] = float(pusher_error)
        metrics[f"state_rmse_{horizon}"] = float(state_rmse)

    return metrics


def select_rollout_cases(data, horizon, num_rollouts, seed):
    rng = np.random.default_rng(seed)
    episode_ids = data["episode_ids"]
    unique_episodes = np.unique(episode_ids)
    rng.shuffle(unique_episodes)

    cases = []
    for episode_id in unique_episodes:
        indices = np.flatnonzero(episode_ids == episode_id)
        if len(indices) <= horizon:
            continue

        max_start = len(indices) - horizon
        start = int(rng.integers(0, max_start + 1))
        rollout_indices = indices[start:start + horizon]

        cases.append(
            {
                "episode_id": int(episode_id),
                "start_step": int(data["step_ids"][indices[start]]),
                "initial_state": data["states"][indices[start]],
                "actions": data["actions"][rollout_indices],
            }
        )

        if len(cases) >= num_rollouts:
            break

    if not cases:
        raise ValueError(f"没有找到长度超过 {horizon} 的轨迹，无法做 rollout 评估。")

    return cases


def plot_rollout(true_states, pred_states, case_name, output_dir, target_radius=TARGET_RADIUS):
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    target_xy = true_states[0, 6:8]
    target_circle = plt.Circle(
        target_xy,
        target_radius,
        fill=False,
        linestyle="--",
        edgecolor="g",
        label="target radius",
    )

    ax.add_patch(target_circle)
    ax.plot(true_states[:, 0], true_states[:, 1], "b-", label="MuJoCo pusher")
    ax.plot(true_states[:, 2], true_states[:, 3], "r-", label="MuJoCo object")
    ax.plot(pred_states[:, 0], pred_states[:, 1], "b--", label="WAM pusher")
    ax.plot(pred_states[:, 2], pred_states[:, 3], "r--", label="WAM object")
    ax.scatter(target_xy[0], target_xy[1], marker="x", c="g", label="target")
    ax.axis("equal")
    ax.grid(True)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(case_name)
    ax.legend()
    fig.tight_layout()

    save_path = os.path.join(output_dir, f"{case_name}.png")
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    return save_path


def aggregate_metrics(all_metrics):
    keys = sorted({key for metrics in all_metrics for key in metrics})
    summary = {}
    for key in keys:
        values = np.asarray([m[key] for m in all_metrics if key in m], dtype=np.float32)
        if len(values) == 0:
            continue
        summary[key] = (float(values.mean()), float(values.std()))
    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate WAM open-loop rollout.")
    parser.add_argument("--checkpoint", default=os.path.join(project_root, "checkpoints", "wam_hard_random_physics_force.pt"))
    parser.add_argument("--data", default=os.path.join(project_root, "data", "trajectories_hard_random_physics.npz"))
    parser.add_argument("--output-dir", default=os.path.join(project_root, "outputs", "rollout_hard_random_physics_force"))
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--num-rollouts", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.checkpoint, device)

    model = WAM(hidden_dim=args.hidden_dim, state_dim=10, action_dim=2).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    state_mean = checkpoint["state_mean"]
    state_std = checkpoint["state_std"]

    data = np.load(args.data)
    cases = select_rollout_cases(data, args.horizon, args.num_rollouts, args.seed)
    env = PushEnv(
        max_step=args.horizon + 5,
        seed=args.seed,
        target_radius=TARGET_RADIUS,
        target_low=HARD_TARGET_LOW,
        target_high=HARD_TARGET_HIGH,
        object_low=HARD_OBJECT_LOW,
        object_high=HARD_OBJECT_HIGH,
        min_object_target_dist=HARD_MIN_OBJECT_TARGET_DIST,
        pusher_backoff=HARD_PUSHER_BACKOFF,
    )

    horizons = [h for h in (10, 20, 50) if h <= args.horizon]
    all_metrics = []

    print(f"device: {device}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"num_rollouts: {len(cases)}")
    print(f"horizons: {horizons}")
    print("-" * 80)

    for rollout_id, case in enumerate(cases):
        true_states = mujoco_rollout(env, case["initial_state"], case["actions"])
        pred_states = wam_rollout(
            model,
            case["initial_state"],
            case["actions"][: len(true_states) - 1],
            state_mean,
            state_std,
            device,
        )

        metrics = compute_metrics(true_states, pred_states, horizons)
        all_metrics.append(metrics)

        case_name = (
            f"rollout_{rollout_id:02d}_ep{case['episode_id']}_step{case['start_step']}"
        )
        plot_path = plot_rollout(true_states, pred_states, case_name, args.output_dir)

        print(case_name)
        for key in sorted(metrics):
            print(f"  {key}: {metrics[key]:.6f}")
        print(f"  plot: {plot_path}")
        print("-" * 80)

    print("summary mean +/- std")
    for key, (mean, std) in aggregate_metrics(all_metrics).items():
        print(f"  {key}: {mean:.6f} +/- {std:.6f}")


if __name__ == "__main__":
    main()
