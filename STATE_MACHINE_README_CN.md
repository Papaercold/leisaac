# 状态机模块：录制与回放说明

## 概述

状态机模块提供无需人工遥操作的自动化数据采集能力。它运行脚本化的策略，将演示数据录制为 HDF5 数据集，并支持回放已录制的演示。

```
scripts/environments/state_machine/
├── pick_orange.py      # 录制 Runner 脚本（拾橙任务）
├── fold_cloth.py       # 录制 Runner 脚本（叠衣任务）
└── replay.py           # 状态机演示回放脚本

source/leisaac/leisaac/state_machine/
├── base.py             # StateMachineBase 抽象基类
├── pick_orange.py      # PickOrangeStateMachine
└── fold_cloth.py       # FoldClothStateMachine
```

---

## 录制

### 快速开始

```bash
# 单臂拾橙（3 个橘子 → 盘子）
python scripts/environments/state_machine/pick_orange.py \
    --task LeIsaac-SO101-PickOrange-v0 \
    --num_envs 1 \
    --record \
    --dataset_file ./datasets/pick_orange.hdf5 \
    --num_demos 10
```

### 工作原理

```
Runner 脚本
  └── gym.make(task, cfg=env_cfg)           # 创建环境
       └── env_cfg.use_teleop_device(       # 配置 IK action manager
               "so101_state_machine")
  └── sm = PickOrangeStateMachine(...)
  └── 主循环:
       actions = sm.get_action(env)         # 状态机计算 8D IK 动作
       env.step(actions)                    # 推进仿真 + 录制器采集数据
       sm.advance()                         # 推进状态机
```

Runner 脚本直接将状态机输出的动作张量传给 `env.step(actions)`，**不经过** `preprocess_device_action()`（后者仅在遥操作流程中使用）。

### 动作格式

| 设备 | 维度 | 格式 |
|---|---|---|
| `so101_state_machine` | 8D | `[pos(3), quat(4), gripper(1)]`，机械臂 base 局部坐标系 |
| `bi_so101_state_machine` | 16D | `[左_pos(3), 左_quat(4), 左_grip(1), 右_pos(3), 右_quat(4), 右_grip(1)]` |

IK 目标位置需表示在**机械臂 base 局部坐标系**下，而非世界坐标系：

```python
diff_w = target_pos_w - base_pos_w
target_pos_local = quat_apply(quat_inv(base_quat_w), diff_w)
```

---

## 数据集结构

演示数据以 HDF5 格式存储在 `data/` 组中：

```
data/
├── demo_0      # 空 episode —— 初始 env.reset() 产生的副产物
│   └── initial_state        # 仅有此字段；num_samples=0；无 actions
├── demo_1      # 第一条真实演示
│   ├── actions              # (T, 8) —— 传入 env.step() 的 IK pose 目标
│   ├── processed_actions    # (T, 6) —— IK 求解器算出的关节目标位置
│   ├── initial_state        # episode 开始时的场景状态（用于 reset_to）
│   ├── states/              # 每步的关节/传感器状态
│   └── obs/                 # 每步的观测（图像、关节位置等）
├── demo_2      # 第二条真实演示
...
```

**重要：** `demo_0` 永远是空的。**第 K 条录制的演示**存储为 `demo_K`。

回放时用 `--select_episodes K` 加载 `demo_K`：

```bash
--select_episodes 1   # → demo_1，第一条真实演示
--select_episodes 2   # → demo_2，第二条真实演示
```

---

## 回放

### 快速开始

```bash
# replay.sh 封装脚本（默认：dataset_test.hdf5 的 demo_3）
bash replay.sh

# 显式调用
python scripts/environments/state_machine/replay.py \
    --task LeIsaac-SO101-PickOrange-v0 \
    --dataset_file ./datasets/pick_orange.hdf5 \
    --task_type so101_state_machine \
    --select_episodes 1 \
    --device cuda \
    --enable_cameras \
    --replay_mode action
```

### 回放模式

| 模式 | 回放的数据 | 适用场景 |
|---|---|---|
| `action` | `HDF5["actions"]`（8D IK pose 目标） | IK 控制设备（`so101_state_machine`） |
| `state` | `HDF5["states"]["articulation"]["robot"]["joint_position"]` | 关节位置控制设备（`so101leader`） |

对 `so101_state_machine` 而言，**只有 `action` 模式有效**。IK action manager 期望 8D pose 输入，而 `state` 模式传入的是 6D 关节位置，会导致维度不匹配。

---

## 技术细节

### 1. 重力禁用（两步缺一不可）

IsaacLab 的 `disable_gravity` 标志只写入关节根 prim，各子 link prim 各自带有 `PhysicsRigidBodyAPI`，重力默认仍然开启。

**第一步** —— 配置层（在 `use_teleop_device()` 中）：
```python
self.scene.robot.spawn.rigid_props.disable_gravity = True
```

**第二步** —— USD Stage 遍历（在 runner 脚本 `gym.make()` 之后）：
```python
_stage = omni.usd.get_context().get_stage()
for _prim in _stage.Traverse():
    if "Robot" in str(_prim.GetPath()) and _prim.HasAPI(UsdPhysics.RigidBodyAPI):
        PhysxSchema.PhysxRigidBodyAPI.Apply(_prim).CreateDisableGravityAttr(True)
```

两步必须同时存在。双臂任务同理（`"Robot"` 可同时匹配 `Left_Robot` 和 `Right_Robot`）。

### 2. 关节阻尼

IK 控制器需要比机械臂默认阻尼（0.6 N·m·s/rad）更高的阻尼值，才能获得稳定、平滑的轨迹。阻尼设置为 **10.0 N·m·s/rad**，在每步调用：

```python
# 在 PickOrangeStateMachine.get_action() 中：
robot = env.scene["robot"]
robot.write_joint_damping_to_sim(damping=10.0)
```

**回放时也必须应用相同阻尼**，以匹配录制条件。状态机回放脚本（`scripts/environments/state_machine/replay.py`）在每次 `env.step()` 前调用 `apply_damping(env, task_type)`。

### 3. 归位 FK 校准

成功判断（`task_done()`）要求关节位于 `SO101_FOLLOWER_REST_POSE_RANGE` 内（例如 shoulder_lift ≈ −100°，并非零位）。由于 IK 控制 EE 位置，无法保证特定关节构型，因此 runner 脚本在调用 `task_done()` 前：

1. 用 `write_joint_state_to_sim()` 将关节传送（teleport）到 rest pose。
2. 只调用 `env.scene.update()`（不调用 `env.sim.step()`），仅刷新数据缓存，避免物理引擎用过期的 actuator 目标把 teleport 撤销。

```python
robot.write_joint_state_to_sim(
    position=_rest_joint_pos,
    velocity=torch.zeros_like(_rest_joint_pos),
)
env.scene.update(dt=env.physics_dt)
success = task_done(env, ...)
```

### 4. 归位策略

放置橘子后，机械臂需要返回 rest pose 才能通过成功判断。

**IK 单独归位不可靠：** 对同一 EE 目标位置，IK 求解器可能找到多个关节解（IK 非唯一性）。从放置后的构型出发，命令 rest pose EE 位置，IK 通常会落入与 rest pose 不同的关节解。

**第三个橘子（最后一个）的当前方案：**

在步骤 620–919 间，关节位置从放置后构型线性插值到 rest pose：

```python
if sm.orange_now == 3 and sm.step_count >= 620:
    if sm.step_count == 620:
        _home_start_pos = _robot.data.joint_pos.clone()
    alpha = (sm.step_count - 620) / 299.0      # 0.0 → 1.0
    blended = _home_start_pos + (_rest_joint_pos - _home_start_pos) * alpha
    _robot.write_joint_state_to_sim(position=blended, velocity=zeros)
# sm.get_action() 的 IK 动作仍然传入 env.step()，用于录制
env.step(actions)
```

**回放局限性：** 录制的 `actions` 是 8D IK pose 目标，而非插值后的关节位置。回放时 IK 求解器从不同的关节起点出发，可能走向不同的路径。HDF5 中的 `processed_actions` 字段存有 IK 实际求解的关节目标位置，但当前回放基础设施使用的是 `actions` 而非 `processed_actions`。

**第一、二个橘子：** 直接用 `sm.advance()` 跳过归位阶段（不调用 `env.step()`），避免在下一个橘子紧随其后时浪费仿真时间。

### 5. IK 动作坐标系

IK 目标必须表示在**机械臂 base 局部坐标系**下，而非世界坐标系：

```python
diff_w        = target_pos_w - robot_base_pos_w
target_pos_lo = quat_apply(quat_inv(robot_base_quat_w), diff_w)
```

### 6. Episode 编号规则

IsaacLab 录制器在第一次 `env.reset()` 调用时（此时还未执行任何步骤）会保存一个仅含初始状态的 episode（`num_samples=0`），即 `demo_0`。

| `--select_episodes N` | 加载的 episode | 内容 |
|---|---|---|
| 0 | `demo_0` | 空（无 actions）—— 会导致 `TypeError` |
| 1 | `demo_1` | 第 1 条真实演示 |
| K | `demo_K` | 第 K 条真实演示 |

---

## 文件说明

| 文件 | 用途 |
|---|---|
| `scripts/environments/state_machine/pick_orange.py` | 拾橙任务录制 Runner |
| `scripts/environments/state_machine/fold_cloth.py` | 叠衣任务录制 Runner |
| `scripts/environments/state_machine/replay.py` | 状态机专用回放脚本（含阻尼设置） |
| `source/leisaac/leisaac/state_machine/base.py` | `StateMachineBase` 抽象基类 |
| `source/leisaac/leisaac/state_machine/pick_orange.py` | `PickOrangeStateMachine` |
| `source/leisaac/leisaac/state_machine/fold_cloth.py` | `FoldClothStateMachine` |
| `replay.sh` | replay.py 的 Shell 封装脚本 |
| `run_task.sh` | 拾橙录制的 Shell 封装脚本 |
