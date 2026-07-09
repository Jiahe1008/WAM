import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, "..")
    data_path = os.path.join(project_root, "data", "trajectories.npz")

    data = np.load(data_path)

    states = data["states"]
    episode_ids = data["episode_ids"]

    unique_episodes = np.unique(episode_ids)

    for eid in unique_episodes[:5]:
        mask = episode_ids == eid
        traj_states = states[mask]

        pusher_xy = traj_states[:, 0:2]
        object_xy = traj_states[:, 2:4]
        target_xy = traj_states[0, 6:8]

        plt.figure()

        plt.plot(pusher_xy[:, 0], pusher_xy[:, 1], label="pusher")
        plt.plot(object_xy[:, 0], object_xy[:, 1], label="object")
        plt.scatter(target_xy[0], target_xy[1], marker="x", label="target")

        plt.title(f"Trajectory {eid}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")
        plt.legend()
        plt.grid(True)
        
        target_radius = 0.16
        circle = plt.Circle(
            target_xy,
            target_radius,
            fill=False,
            linestyle="--",
            label="target radius"
        )

        plt.gca().add_patch(circle)
    plt.show()


if __name__ == "__main__":
    main()