import os
import numpy as np


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, "..")
    data_path = os.path.join(project_root, "data", "trajectories_hard_random_physics.npz")

    data = np.load(data_path)

    states = data["states"]
    actions = data["actions"]
    next_states = data["next_states"]
    rewards = data["rewards"]
    dones = data["dones"]
    episode_ids = data["episode_ids"]
    step_ids = data["step_ids"]
    instruction_ids = data["instruction_ids"]

    print("states:", states.shape)
    print("actions:", actions.shape)
    print("next_states:", next_states.shape)
    print("rewards:", rewards.shape)
    print("dones:", dones.shape)
    print("episode_ids:", episode_ids.shape)
    print("step_ids:", step_ids.shape)
    print("instruction_ids:", instruction_ids.shape)

    print("-" * 50)

    unique_episodes = np.unique(episode_ids)
    print("num episodes:", len(unique_episodes))
    print("num transitions:", len(states))

    lengths = []
    for eid in unique_episodes:
        lengths.append(np.sum(episode_ids == eid))

    lengths = np.array(lengths)

    print("trajectory length:")
    print("  min:", lengths.min())
    print("  max:", lengths.max())
    print("  mean:", lengths.mean())

    print("-" * 50)

    print("state sample:")
    print(states[0])

    print("action sample:")
    print(actions[0])

    print("next_state sample:")
    print(next_states[0])

    print("-" * 50)

    print("state min:", states.min(axis=0))
    print("state max:", states.max(axis=0))
    object_target_dist = np.linalg.norm(states[:, 2:4] - states[:, 6:8], axis=1)
    print("object-target distance:")
    print("  min:", object_target_dist.min())
    print("  max:", object_target_dist.max())
    print("  mean:", object_target_dist.mean())

    print("action min:", actions.min(axis=0))
    print("action max:", actions.max(axis=0))


if __name__ == "__main__":
    main()
