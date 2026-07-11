'''
training data format
{
    "state": state_t,
    "instruction_id": 0,
    "action": action_t,
    "next_state": next_state_t,
    "reward": reward_t,
    "done": done_t,
    "episode_id": episode_id,
    "step_id": step_id,
}
'''




import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)
import numpy as np
from env.push_env import PushEnv

all_states = []
all_actions = []
all_next_states = []
all_rewards = []
all_dones = []
all_episode_ids = []
all_step_ids = []
all_instruction_ids = []
MAX_STEP = 1000
SAVED_EPISODE = 800
saved_episode_id = 0
env = PushEnv(seed=0)

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


def run():
    global saved_episode_id
    episode_data = []
    state = env.reset()
    for step in range(MAX_STEP):
        action = expert_action(state)

        # (10,), float, bool, {"success":bool, "distance":float}
        noise = np.random.normal(0.0, 0.04, size=2)
        action = np.clip(action + noise, -1.0, 1.0)
        next_state, reward, done, info = env.step(action)  
        episode_data.append((state, action, next_state, reward, done, step))
        state = next_state

        if done:
            break
    
    if info["success"]:
        for state, action, next_state, reward, done, step_id in episode_data:
            all_states.append(state)
            all_actions.append(action)
            all_next_states.append(next_state)
            all_rewards.append(reward)
            all_dones.append(done)
            all_episode_ids.append(saved_episode_id)
            all_step_ids.append(step_id)
            all_instruction_ids.append(0)

        print(f"saved episode {saved_episode_id}, length={len(episode_data)}")
        saved_episode_id += 1
    else:
        print(f"failed episode, length={len(episode_data)}, final_dist={info['distance']:.3f}")
        

def main():
    print(f"collecting data of {SAVED_EPISODE} trajetories")
    while(saved_episode_id < SAVED_EPISODE):
        run()
    
    states = np.array(all_states, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.float32)
    next_states = np.array(all_next_states, dtype=np.float32)
    rewards = np.array(all_rewards, dtype=np.float32)
    dones = np.array(all_dones, dtype=np.bool_)
    episode_ids = np.array(all_episode_ids, dtype=np.int64)
    step_ids = np.array(all_step_ids, dtype=np.int64)
    instruction_ids = np.array(all_instruction_ids, dtype=np.int64)

    save_dir = os.path.join(project_root, "data")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "trajectories.npz")

    np.savez_compressed(
        save_path,
        states=states,
        actions=actions,
        next_states=next_states,
        rewards=rewards,
        dones=dones,
        episode_ids=episode_ids,
        step_ids=step_ids,
        instruction_ids=instruction_ids,
    )

    print("saved data to ../data/trajectories.npz")
    print("states:", states.shape)
    print("actions:", actions.shape)
    print("next_states:", next_states.shape)
    print("episodes:", len(np.unique(episode_ids)))

if __name__ == "__main__":
    main()