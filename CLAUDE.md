# LeIsaac Project — Claude Context

## 项目结构概览

```
leisaac/
├── scripts/environments/state_machine/   # 运行脚本
│   ├── pick_orange.py                    # 单臂拾橙数据采集脚本
│   └── fold_cloth.py                     # 双臂叠衣数据采集脚本（新增）
└── source/leisaac/leisaac/
    ├── state_machine/                    # 状态机模块（新增）
    │   ├── __init__.py
    │   ├── base.py                       # StateMachineBase 抽象基类
    │   ├── pick_orange.py                # PickOrangeStateMachine
    │   └── fold_cloth.py                 # FoldClothStateMachine（新增）
    ├── devices/
    │   └── action_process.py             # 设备动作配置（新增 bi_so101_state_machine）
    ├── tasks/
    │   ├── template/
    │   │   ├── bi_arm_env_cfg.py         # 双臂 ManagerBased env 基础配置
    │   │   ├── single_arm_env_cfg.py     # 单臂 ManagerBased env 基础配置
    │   │   └── direct/
    │   │       └── bi_arm_env.py         # 双臂 Direct env 基础配置
    │   ├── pick_orange/                  # 单臂拾橙任务
    │   │   └── mdp/terminations.py       # task_done() 检查 rest pose + orange in plate
    │   └── fold_cloth/                   # 双臂叠衣任务
    │       ├── mdp/terminations.py       # cloth_folded() 检查 rest pose + 布料关键点距离
    │       ├── fold_cloth_bi_arm_env_cfg.py  # ManagerBased env 配置（含 cloths 场景元素）
    │       └── direct/
    │           └── fold_cloth_bi_arm_env.py  # Direct env（含 ClothObject 初始化）
    └── utils/
        ├── env_utils.py                  # dynamic_reset_gripper_effort_limit_sim 等工具
        └── robot_utils.py                # is_so101_at_rest_pose, convert_leisaac_action_to_lerobot
```

---

## 状态机架构

### 状态机与环境的关系

```
Runner script
  └── gym.make(task_name, cfg=env_cfg)        # 创建环境
       └── env_cfg.use_teleop_device(device)  # 配置 action manager
  └── sm = StateMachine(...)
  └── 主循环:
       actions = sm.get_action(env)           # 状态机计算动作
       env.step(actions)                      # 直接传入 tensor，不经过 preprocess_device_action
       sm.advance()
```

**关键点**：runner script 直接调用 `env.step(actions)` 传入动作 tensor，**不经过**
`preprocess_device_action()`（后者只在遥操作流程中使用）。

### 设备类型与 action 格式

| device | action 维度 | 格式 | 适用场景 |
|---|---|---|---|
| `so101_state_machine` | 8D | `[pos(3), quat(4), gripper(1)]` | 单臂 IK 控制 |
| `bi_so101_state_machine` | 16D | `[left_pos(3), left_quat(4), left_grip(1), right_pos(3), right_quat(4), right_grip(1)]` | 双臂 IK 控制 |

IK 目标位置为**机械臂 base frame 下的局部坐标**，需要从世界坐标转换：
```python
diff_w = target_pos_w - base_pos_w
target_pos_local = quat_apply(quat_inv(base_quat_w), diff_w)
```

---

## 关键技术细节

### 1. 重力禁用（两步缺一不可）

**第一步**：`use_teleop_device()` 中设置 rigid_props（只禁用 root prim）：
```python
# bi_arm_env_cfg.py / bi_arm_env.py（Direct）
if teleop_device in ["bi_so101_state_machine"]:
    self.scene.left_arm.spawn.rigid_props.disable_gravity = True
    self.scene.right_arm.spawn.rigid_props.disable_gravity = True
```

**第二步**：runner script 中 USD stage 遍历（禁用所有子 link prim）：
```python
import omni.usd
from pxr import PhysxSchema, UsdPhysics
_stage = omni.usd.get_context().get_stage()
for _prim in _stage.Traverse():
    if "Robot" in str(_prim.GetPath()) and _prim.HasAPI(UsdPhysics.RigidBodyAPI):
        PhysxSchema.PhysxRigidBodyAPI.Apply(_prim).CreateDisableGravityAttr(True)
```

- 单臂场景 prim 路径含 `Robot`
- 双臂场景 prim 路径含 `Left_Robot` 和 `Right_Robot`，均可被 `"Robot" in path` 匹配

### 2. FK 校准（获取 rest pose 的 EE 世界坐标）

`task_done()` / `cloth_folded()` 要求机械臂关节在 `SO101_FOLLOWER_REST_POSE_RANGE` 内：
- shoulder_lift ≈ -100°，elbow_flex ≈ 90°，wrist_flex ≈ 50°（并非零位）

IK 控制 EE 位置，无法直接保证关节角度，因此需要 FK 校准：先把关节
teleport 到 rest pose，读取此时的 EE 世界坐标，再让状态机在 return home
阶段命令 IK 趋向该位置（给足够步数收敛，约 300 步 / 5 秒）。

```python
# 正确 API（teleport 关节到指定位置）
robot.write_joint_state_to_sim(
    position=_rest_joint_pos,          # (num_envs, num_joints)
    velocity=torch.zeros_like(...),
)
env.sim.step(render=False)
env.scene.update(dt=env.physics_dt)
rest_ee_pos = robot.data.body_pos_w[:, -1, :].clone()  # gripper EE 世界坐标
```

> ⚠️ `write_joint_position_target_to_sim` 不存在，正确方法是 `write_joint_state_to_sim`。

### 3. ClothObject 在 ManagerBased env 中的初始化

`FoldClothBiArmEnv`（Direct env）会自动初始化 ClothObject，但 ManagerBased env
不会。runner script 中需手动添加：
```python
from leisaac.enhance.assets import ClothObject
if not hasattr(env.scene, "particle_objects"):
    env.scene.particle_objects = {}
env.scene.particle_objects["cloths"] = ClothObject(
    cfg=env.cfg.scene.cloths,
    scene=env.scene,
)
env.scene.particle_objects["cloths"].initialize()
```

每次 episode 重置时，需在 `env.reset()` **之前**先 reset cloth（与 Direct env 一致）：
```python
env.scene.particle_objects["cloths"].reset()
env.reset()
```

### 4. 布料关键点

```python
_CLOTH_KEYPOINTS = [159789, 120788, 115370, 159716, 121443, 112382]
# 顺序: left_sleeve, left_shoulder, left_hem, right_sleeve, right_shoulder, right_hem

cloth = env.scene.particle_objects["cloths"]
kp = cloth.point_positions[:, _CLOTH_KEYPOINTS, :]  # (num_envs, 6, 3) 世界坐标
```

`cloth_folded()` 检查的距离条件（threshold=0.20m）：
- left_sleeve → right_shoulder
- right_sleeve → left_shoulder
- left_hem → left_shoulder
- right_hem → right_shoulder

---

## 本次会话新增/修改的文件

| 文件 | 类型 | 内容 |
|---|---|---|
| `state_machine/fold_cloth.py` | 新建 | `FoldClothStateMachine`，1260 步，13 个折叠阶段 |
| `state_machine/__init__.py` | 修改 | 新增 `FoldClothStateMachine` 导出 |
| `devices/action_process.py` | 修改 | 新增 `bi_so101_state_machine` 设备分支（16D IK） |
| `tasks/template/bi_arm_env_cfg.py` | 修改 | `use_teleop_device()` 新增 `bi_so101_state_machine` 的重力禁用 |
| `tasks/template/direct/bi_arm_env.py` | 修改 | 同上（Direct env 版本） |
| `scripts/.../state_machine/fold_cloth.py` | 新建 | fold_cloth 数据采集 runner script |

### 前一次会话（pick_orange）修改的文件

| 文件 | 内容 |
|---|---|
| `state_machine/base.py` | `StateMachineBase` 抽象基类 |
| `state_machine/pick_orange.py` | `PickOrangeStateMachine`，含 rest pose 回位阶段 |
| `scripts/.../state_machine/pick_orange.py` | 重力禁用 + FK 校准 + rest_ee_pos_world 传参 |

---

## 任务注册名

| gym 注册 ID | 说明 |
|---|---|
| `LeIsaac-SO101-PickOrange-v0` | 单臂拾橙，ManagerBased |
| `LeIsaac-SO101-FoldCloth-BiArm-v0` | 双臂叠衣，ManagerBased（推荐用于状态机，有 IK action manager） |
| `LeIsaac-SO101-FoldCloth-BiArm-Direct-v0` | 双臂叠衣，Direct env（action space = 12D 关节位置） |

---

## 运行方式

```bash
# pick_orange（单臂）
python scripts/environments/state_machine/pick_orange.py \
    --task LeIsaac-SO101-PickOrange-v0 --num_envs 1 --record \
    --dataset_file ./datasets/pick_orange.hdf5 --num_demos 10

# fold_cloth（双臂）
python scripts/environments/state_machine/fold_cloth.py \
    --task LeIsaac-SO101-FoldCloth-BiArm-v0 --num_envs 1 --record \
    --dataset_file ./datasets/fold_cloth.hdf5 --num_demos 10
```

---

## 注意事项

- `preprocess_device_action()` 在 runner script 流程中**不会被调用**，无需为
  `bi_so101_state_machine` 添加对应分支（除非日后接入真实设备遥操作）
- `env_utils.py:dynamic_reset_gripper_effort_limit_sim()` 对 fold_cloth 无效
  （`dynamic_reset_gripper_effort_limit = False`），暂无需修改
- fold_cloth 的 `decimation = 2`，实际物理步频为 `sim_dt / 2`，注意时序计算
- 状态机的步数常量（`MAX_STEPS`、各阶段边界）均基于 60 Hz 控制频率设计
