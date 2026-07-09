import os
import numpy as np
import mujoco

class PushEnv:
    def __init__(self, xml_path=None, max_step=300, seed=42):
        if xml_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            xml_path = os.path.join(current_dir, "push_env.xml")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.max_steps = max_step
        self.step_count = 0
        self.rng = np.random.default_rng(seed)

        self.target_radius = 0.16
        self.target_xy = np.array([0.8, 0.5], dtype=np.float32)

        self.pusher_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pusher")
        self.object_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        self.object_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_free")

        self.object_qpos_adr = self.model.jnt_qposadr[self.object_joint_id]
        self.object_qvel_adr = self.model.jnt_dofadr[self.object_joint_id]
        self.pusher_mocap_id = self.model.body_mocapid[self.pusher_body_id]

        self.current_mass = 0.2
        self.current_friction = 0.6

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0

        self.target_xy = self.rng.uniform(
            low=np.array([0.45, -0.6]),high=np.array([0.95, 0.6])).astype(np.float32)

        object_xy = self.rng.uniform(
            low=np.array([-0.3, -0.5]),high=np.array([0.2, 0.5])).astype(np.float32)
        
        push_dir = self.target_xy - object_xy
        push_dir = push_dir / (np.linalg.norm(push_dir) + 1e-6)

        pusher_xy = object_xy - push_dir * 0.25

        qpos_adr = self.object_qpos_adr
        self.data.qpos[qpos_adr + 0] = object_xy[0]
        self.data.qpos[qpos_adr + 1] = object_xy[1]
        self.data.qpos[qpos_adr + 2] = 0.05
        self.data.qpos[qpos_adr + 3] = 1.0
        self.data.qpos[qpos_adr + 4] = 0.0
        self.data.qpos[qpos_adr + 5] = 0.0
        self.data.qpos[qpos_adr + 6] = 0.0

        qvel_adr = self.object_qvel_adr
        self.data.qvel[qvel_adr:qvel_adr + 6] = 0.0

        self.data.mocap_pos[self.pusher_mocap_id][0] = pusher_xy[0]
        self.data.mocap_pos[self.pusher_mocap_id][1] = pusher_xy[1]
        self.data.mocap_pos[self.pusher_mocap_id][2] = 0.07

        mujoco.mj_forward(self.model, self.data)

        return self.get_state()
    
    def step(self, action):
        self.step_count += 1

        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        delta_xy = action * 0.025
        pusher_pos = self.data.mocap_pos[self.pusher_mocap_id].copy() #[x, y, z]
        new_xy = pusher_pos[:2] + delta_xy
        new_xy = np.clip(new_xy, -1.2, 1.2)

        self.data.mocap_pos[self.pusher_mocap_id][0] = new_xy[0]
        self.data.mocap_pos[self.pusher_mocap_id][1] = new_xy[1]
        self.data.mocap_pos[self.pusher_mocap_id][2] = 0.07

        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
        
        next_state = self.get_state()
        reward = self.compute_reward()
        done = self.is_done()

        info = {
            "success": self.is_success(),
            "distance": self.distance_to_target(),
        }

        return next_state, reward, done, info
    
    def get_state(self):
        pusher_xy = self.data.mocap_pos[self.pusher_mocap_id][:2].copy()

        object_pos = self.data.xpos[self.object_body_id].copy()
        object_xy = object_pos[:2]

        qvel_adr = self.object_qvel_adr
        object_vxy = self.data.qvel[qvel_adr:qvel_adr + 2].copy()

        state = np.array([
            pusher_xy[0],
            pusher_xy[1],
            object_xy[0],
            object_xy[1],
            object_vxy[0],
            object_vxy[1],
            self.target_xy[0],
            self.target_xy[1],
            self.current_mass,
            self.current_friction,
        ], dtype=np.float32)

        return state

    def distance_to_target(self):
        object_xy = self.data.xpos[self.object_body_id][:2].copy()
        return float(np.linalg.norm(object_xy - self.target_xy))

    def is_success(self):
        return self.distance_to_target() < self.target_radius

    def is_done(self):
        if self.is_success():
            return True

        if self.step_count >= self.max_steps:
            return True

        return False

    def compute_reward(self):
        dist = self.distance_to_target()
        reward = -dist

        if self.is_success():
            reward += 10.0

        return float(reward)