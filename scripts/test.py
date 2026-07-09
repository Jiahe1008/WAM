import os
import sys
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)

from env.push_env import PushEnv


def expert_action(state):
    """启发式专家策略：先到物体后方，再推向目标"""
    pusher_xy = state[0:2]
    object_xy = state[2:4]
    target_xy = state[6:8]

    push_dir = target_xy - object_xy
    push_dir = push_dir / (np.linalg.norm(push_dir) + 1e-6)

    desired_pusher_xy = object_xy - push_dir * 0.18
    to_desired = desired_pusher_xy - pusher_xy

    if np.linalg.norm(to_desired) > 0.05:
        action = to_desired / (np.linalg.norm(to_desired) + 1e-6)
    else:
        action = push_dir

    return action.astype(np.float32)


def main():
    env = PushEnv(seed=0)
    state = env.reset()

    for step in range(300):
        action = expert_action(state)
        next_state, reward, done, info = env.step(action)

        if step % 20 == 0:
            print(
                f"step={step:03d} | "
                f"dist={info['distance']:.3f} | "
                f"reward={reward:.3f} | "
                f"success={info['success']}"
            )

        state = next_state

        if done:
            print("done")
            print("success:", info["success"])
            print("final distance:", info["distance"])
            break


if __name__ == "__main__":
    main()