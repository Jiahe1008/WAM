# 从 VLA 到 WAM

本项目在 MuJoCo 平面推物任务中实现并比较两类模型：

- **VLA baseline**：从当前状态直接预测动作，只进行行为模仿。
- **WAM（World Action Model）**：共享状态编码器，同时预测动作、下一时刻状态和接触力，并通过物理一致性损失约束 world head。

项目按照课程任务的四个阶段展开：

1. 搭建可调质量和摩擦系数的 MuJoCo 交互环境，采集专家轨迹。
2. 实现共享表征的 WAM，并联合训练 action head、world head 和 force head。
3. 通过 10/20/50 步开环 rollout 验证未来状态预测和物理一致性。
4. 进行 WAM、VLA 和参数量匹配 VLA 的闭环控制与物理泛化对比。

## 当前结论

早期简单任务中，WAM 和 VLA 都能较容易完成任务；使用固定质量和摩擦训练时，两者在变化物理参数下都会严重失效。为此，本项目依次加入：

- 每条轨迹随机采样物体质量和摩擦系数；
- 更小的目标半径、更远的物体—目标距离和更大的位置范围；
- 与 WAM 参数量接近的 `VLA-matched` 公平基线；
- 接触力预测头和自监督动力学残差。

最终高难度随机物理实验中，三种模型的参数量为：

| 模型 | hidden dim | 参数量 |
| --- | ---: | ---: |
| VLA | 32 | 1,474 |
| VLA-matched | 65 | 5,137 |
| WAM | 32 | 5,166 |

闭环成功率如下，每种工况测试 100 个相同初始状态：

| 物理工况 | WAM | VLA | VLA-matched |
| --- | ---: | ---: | ---: |
| train：`m=0.2, μ=0.6` | **96%** | 91% | 92% |
| heavy：`m=0.4, μ=0.6` | **93%** | 88% | 85% |
| low friction：`m=0.2, μ=0.25` | **96%** | 90% | 88% |
| heavy + low friction：`m=0.4, μ=0.25` | **95%** | 85% | 80% |

WAM 在四种工况中的平均成功率为 94.5%，相对参数量匹配 VLA 提高 8.2 个百分点。50 步开环预测仍存在明显误差累积，因此当前结果支持的是“物理辅助任务改善了短期动力学表征和闭环鲁棒性”，而不是已经获得通用物理推理能力。

完整分析见 [实验报告](reports/WAM实验报告_从VLA到WAM.docx)。

## 项目结构

```text
env/
  push_env.xml                     MuJoCo 场景
  push_env.py                      环境、状态、奖励和物理参数设置
data/
  dataset.py                       VLA/WAM Dataset
  trajectories_*.npz              不同阶段采集的轨迹
model/
  VLA.py                           纯动作预测基线
  WAM.py                           共享编码器和三个预测头
train/
  physical_loss.py                 物理约束与动力学残差
  train_vla.py                     VLA 和 VLA-matched 训练
  train_wam.py                     WAM 联合训练
scripts/
  collect_data.py                  专家轨迹采集
  inspect_data.py                  数据统计
  plot_data.py                     轨迹可视化
  eval_rollout.py                  多步开环评估
  eval_closed_loop.py              闭环控制与泛化评估
checkpoints/                       best model 和 JSON 训练历史
outputs/                           rollout 图片和闭环 CSV
reports/                           Word 实验报告及生成脚本
```

## 环境安装

项目使用 Python、PyTorch、MuJoCo、NumPy 和 Matplotlib。已有 Conda 环境时：

```bash
conda activate wam
python -m pip install -r requirements.txt
```

检查 PyTorch 是否能使用 GPU：

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

训练和评估脚本会自动选择 `cuda`；CUDA 不可用时回退到 CPU。

## 任务与数据

### 环境状态

每个状态是 10 维向量：

```text
[pusher_x, pusher_y,
 object_x, object_y,
 object_vx, object_vy,
 target_x, target_y,
 object_mass, object_friction]
```

动作是推杆在平面内的二维移动方向，范围为 `[-1, 1]`。当前实验只有一条固定任务指令，使用 `instruction_id=0` 表示“将物体推入目标区域”。

### 高难度随机物理设置

当前默认实验使用：

| 配置 | 范围 |
| --- | --- |
| 目标半径 | `0.10` |
| 质量 | `[0.10, 0.50]` |
| 摩擦系数 | `[0.25, 0.90]` |
| 目标位置 | `x∈[0.55,1.05], y∈[-0.95,0.95]` |
| 物体位置 | `x∈[-0.85,0.10], y∈[-0.85,0.85]` |
| 初始物体—目标距离 | 不小于 `0.75` |
| 有效轨迹数 | `800` |
| 转移数 | `91,978` |

专家策略先移动到物体后方，再沿目标方向推动，并在动作上加入标准差为 `0.04` 的高斯噪声。只保存最终成功的 episode。

### 数据格式

`data/trajectories_hard_random_physics.npz` 包含：

```text
states:          (N, 10)
actions:         (N, 2)
next_states:     (N, 10)
rewards:         (N,)
dones:           (N,)
episode_ids:     (N,)
step_ids:        (N,)
instruction_ids: (N,)
```

采集、检查和可视化：

```bash
python scripts/collect_data.py
python scripts/inspect_data.py
python scripts/plot_data.py
```

`collect_data.py` 默认写入：

```text
data/trajectories_hard_random_physics.npz
```

若文件已经存在，采集脚本会拒绝覆盖。重新采集前应修改输出文件名或备份原文件。

## 模型

### VLA baseline

VLA 使用两层 MLP：

```text
state -> shared MLP -> action
```

训练目标只有动作均方误差：

```text
L_vla = MSE(pred_action, expert_action)
```

### WAM

WAM 使用一个共享编码器和三个预测头：

```text
state -> encoder -> action head -> pred_action
                 -> world head  -> pred_next_state
                 -> force head  -> pred_contact_force
```

world head 接收共享隐状态和当前动作，以残差形式预测下一状态：

```text
pred_next_state = state + delta_state
```

force head 预测二维接触力。它不直接拟合 MuJoCo 的接触力标签，而是通过真实速度变化构造自监督牛顿动力学残差。

## 物理约束

物理损失集中在 [physical_loss.py](train/physical_loss.py)：

1. **无接触无源增能**：当前和预测下一状态均无接触时，物体动能不应无源增加。
2. **冲量方向一致**：接触状态下，速度变化方向应与推杆到物体的接触法向一致。
3. **刚体不可重叠**：惩罚预测的推杆和物体几何穿透。
4. **动力学残差**：预测接触力应解释数据中的真实速度变化。

动力学残差近似为：

```text
m * (v_next - v_cur) / dt = F_contact_pred + F_friction
```

WAM 总损失：

```text
L = L_action + 1.0 * L_pred + 0.01 * L_phys + 0.001 * L_force
```

物理项权重较小是因为各损失的量纲和数值尺度不同。训练日志同时记录原始损失和加权后的实际贡献。

## 训练

默认配置：

```text
optimizer: AdamW
learning rate: 1e-3
batch size: 128
epochs: 100
train/validation split: 80% / 20%，按 episode 划分
checkpoint policy: 保存验证损失最低的 best model
```

推荐依次运行：

```bash
conda activate wam

python train/train_vla.py
python train/train_vla.py --matched
python train/train_wam.py
```

输出文件：

```text
checkpoints/vla_hard_random_physics.pt
checkpoints/vla_hard_random_physics_history.json

checkpoints/vla_matched_hard_random_physics.pt
checkpoints/vla_matched_hard_random_physics_history.json

checkpoints/wam_hard_random_physics_force.pt
checkpoints/wam_hard_random_physics_force_history.json
```

三次训练均保存 best model，而不是最后一个 epoch。若目标 checkpoint 或 history 已存在，脚本会拒绝覆盖。

VLA 也支持自定义参数：

```bash
python train/train_vla.py \
  --data-filename trajectories_hard_random_physics.npz \
  --hidden-dim 32 \
  --num-epochs 100 \
  --checkpoint-filename my_vla.pt \
  --history-filename my_vla_history.json
```

## 任务 3：多步开环 Rollout

开环评估只在第 0 步使用真实状态。之后 WAM 每一步使用自己预测的状态，并与相同初始条件、相同动作序列下的 MuJoCo 轨迹比较：

```bash
python scripts/eval_rollout.py
```

默认配置：

```text
checkpoint: checkpoints/wam_hard_random_physics_force.pt
data: data/trajectories_hard_random_physics.npz
rollouts: 5
horizon: 50
metrics: 10/20/50 步位置误差、状态 RMSE、穿透率、穿透深度、速度跳变
output: outputs/rollout_hard_random_physics_force/
```

当前 5 条轨迹的汇总结果：

| Horizon | 物体位置误差 | 推杆位置误差 | 状态 RMSE |
| ---: | ---: | ---: | ---: |
| 10 | `0.0189 ± 0.0099` | `0.0259 ± 0.0145` | `0.0182 ± 0.0038` |
| 20 | `0.0462 ± 0.0096` | `0.0447 ± 0.0227` | `0.0406 ± 0.0190` |
| 50 | `0.3180 ± 0.2726` | `0.1961 ± 0.0731` | `0.1525 ± 0.0747` |

平均预测穿透深度为 `0.00271`。50 步误差明显增大，说明单步 world head 自回归时仍有长期误差累积。

自定义评估：

```bash
python scripts/eval_rollout.py \
  --horizon 20 \
  --num-rollouts 10 \
  --output-dir outputs/rollout_custom
```

同一输出目录下的同名 rollout 图片会被重新写入，保留旧结果时应指定新的 `--output-dir`。

## 任务 4：闭环控制

闭环评估每一步都重新读取 MuJoCo 当前状态，再由模型动作头输出动作：

```bash
python scripts/eval_closed_loop.py
```

默认比较：

- `WAM`：5,166 个参数；
- `VLA`：1,474 个参数；
- `VLA-matched`：5,137 个参数，checkpoint 存在时自动加入。

默认物理工况：

```text
train:                mass=0.2, friction=0.6
heavy:                mass=0.4, friction=0.6
low_friction:         mass=0.2, friction=0.25
heavy_low_friction:   mass=0.4, friction=0.25
```

每种工况运行 100 个 episode，最大 250 步，输出动作 MSE、成功率、平均步数、最终距离、最小距离和累计奖励：

```text
outputs/closed_loop_hard_random_physics_force/closed_loop_episodes.csv
outputs/closed_loop_hard_random_physics_force/closed_loop_summary.csv
```

自定义测试工况时应使用新的输出目录：

```bash
python scripts/eval_closed_loop.py \
  --episodes 100 \
  --output-dir outputs/closed_loop_custom \
  --physics-cases "light:0.1:0.6,very_heavy:0.5:0.6,slippery:0.2:0.15"
```

闭环脚本在结果 CSV 已存在时会拒绝覆盖。

## 完整复现顺序

若从零开始复现实验，先确保默认数据和输出路径没有同名文件：

```bash
conda activate wam

python scripts/collect_data.py
python scripts/inspect_data.py

python train/train_vla.py
python train/train_vla.py --matched
python train/train_wam.py

python scripts/eval_rollout.py
python scripts/eval_closed_loop.py
```

数据采集和三次训练耗时较长；评估前可先使用仓库中已有的数据和 checkpoint。

## 实验报告

Word 报告位于：

```text
reports/WAM实验报告_从VLA到WAM.docx
```

报告生成脚本读取现有 JSON、CSV 和 rollout 图片，不会重新训练模型：

```bash
python reports/build_report.py
```

为避免覆盖已修改的报告，若目标 Word 文件已经存在，生成脚本会直接报错。需要重新生成时，请先将旧报告改名。

## 当前局限

- 当前输入是低维状态，不是原始图像。
- 语言指令被简化为单一 `instruction_id`。
- world head 是确定性单步 MLP，没有 GRU/RSSM 的历史记忆。
- 接触力是自监督隐变量，尚未与 MuJoCo 真实接触力做独立标定。
- 当前环境只有单物体二维平移，没有姿态、力矩和多物体碰撞。
- 尚未实现 WAM-MPC；当前任务四使用 action head 直接闭环控制。

可继续扩展多步训练、GRU/RSSM、物理损失消融、真实接触力监督以及基于 world head 的 MPC 规划。
