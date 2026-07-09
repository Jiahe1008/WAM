import os
import time
import numpy as np
import mujoco
import mujoco.viewer


def get_body_pos(model, data, body_name):
    """读取 body 的世界坐标"""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return data.xpos[body_id].copy()


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path =  "/home/jiahe/code/lab/wam/env/push_env.xml"

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    target_xy = np.array([0.8, 0.5])
    pusher_mocap_id = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0

        while viewer.is_running():
            object_pos = get_body_pos(model, data, "object")  # object 是红圆柱的位置
            pusher_pos = data.mocap_pos[pusher_mocap_id].copy()  # pusher是蓝色球杆的位置。[x,y, z]

            object_xy = object_pos[:2]  # [x, y]
            pusher_xy = pusher_pos[:2]

            push_dir = target_xy - object_xy
            push_dir = push_dir / (np.linalg.norm(push_dir) + 1e-6) # np.linalg.norm计算向量长度（L2范数）

            desired_pusher_xy = object_xy - push_dir * 0.14  # 篮球应该在红圆柱体后面
            to_desired = desired_pusher_xy - pusher_xy

            if np.linalg.norm(to_desired) > 0.03:  # 启发式规则，先跑到红球后面，然后推着红球往target跑
                move_dir = to_desired / (np.linalg.norm(to_desired) + 1e-6)
            else:
                move_dir = push_dir

            speed = 0.006
            new_pusher_xy = pusher_xy + move_dir * speed
            new_pusher_xy = np.clip(new_pusher_xy, -1.2, 1.2)

            data.mocap_pos[pusher_mocap_id][0] = new_pusher_xy[0]  # 直接控制球位置
            data.mocap_pos[pusher_mocap_id][1] = new_pusher_xy[1]
            data.mocap_pos[pusher_mocap_id][2] = 0.07

            mujoco.mj_step(model, data)
            viewer.sync()

            time.sleep(model.opt.timestep)
            step += 1


if __name__ == "__main__":
    main()
    