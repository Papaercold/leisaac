"""Script to generate data using state machine with leisaac fold-cloth bi-arm environment.

Launch Isaac Sim Simulator first.

Default task: LeIsaac-SO101-FoldCloth-BiArm-v0 (manager-based, bi-arm IK control)
"""

import argparse
import multiprocessing
import os
import time

if multiprocessing.get_start_method() != 'spawn':
    multiprocessing.set_start_method('spawn', force=True)

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description='leisaac data generation script for fold_cloth task.')
parser.add_argument('--num_envs', type=int, default=1)
parser.add_argument('--task', type=str, default=None)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--record', action='store_true')
parser.add_argument('--step_hz', type=int, default=60)
parser.add_argument('--dataset_file', type=str, default='./datasets/dataset.hdf5')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--num_demos', type=int, default=1)
parser.add_argument('--quality', action='store_true')
parser.add_argument('--use_lerobot_recorder', action='store_true', help='whether to use lerobot recorder.')
parser.add_argument('--lerobot_dataset_repo_id', type=str, default=None, help='Lerobot Dataset repository ID.')
parser.add_argument('--lerobot_dataset_fps', type=int, default=30, help='Lerobot Dataset frames per second.')

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher_args = vars(args_cli)

app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import DatasetExportMode, SceneEntityCfg, TerminationTermCfg
from isaaclab_tasks.utils import parse_env_cfg

from leisaac.enhance.assets import ClothObject
from leisaac.enhance.managers import EnhanceDatasetExportMode, StreamingRecorderManager
from leisaac.state_machine import FoldClothStateMachine
from leisaac.tasks.fold_cloth.mdp import cloth_folded
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def auto_terminate(env: ManagerBasedRLEnv | DirectRLEnv, success: bool):
    """Programmatically mark the current episode as success or failure."""
    if hasattr(env, "termination_manager"):
        if success:
            env.termination_manager.set_term_cfg(
                "success",
                TerminationTermCfg(func=lambda env: torch.ones(env.num_envs, dtype=torch.bool, device=env.device)),
            )
        else:
            env.termination_manager.set_term_cfg(
                "success",
                TerminationTermCfg(func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)),
            )
        env.termination_manager.compute()
    elif hasattr(env, "_get_dones"):
        env.cfg.return_success_status = success
    return False


def main() -> None:
    """Run a fold-cloth state machine in an Isaac Lab bi-arm manipulation environment.

    Uses ``LeIsaac-SO101-FoldCloth-BiArm-v0`` (manager-based) with the
    ``bi_so101_state_machine`` device so that both arms are controlled via
    absolute-pose IK.  A ``ClothObject`` is initialised manually on the scene
    so that the state machine can read cloth particle keypoints at runtime.
    """
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    task_name = args_cli.task if args_cli.task is not None else "LeIsaac-SO101-FoldCloth-BiArm-v0"
    env_cfg = parse_env_cfg(task_name, device=args_cli.device, num_envs=args_cli.num_envs)

    # Bi-arm IK state-machine device
    device = "bi_so101_state_machine"
    env_cfg.use_teleop_device(device)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())

    # Timeout and termination preprocessing
    is_direct_env = "Direct" in task_name
    if is_direct_env:
        env_cfg.never_time_out = True
        env_cfg.auto_terminate = True
    else:
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
        if hasattr(env_cfg.terminations, "success"):
            env_cfg.terminations.success = None

    # Recorder preprocessing & manual success-termination preprocessing
    if args_cli.record:
        if args_cli.use_lerobot_recorder:
            if args_cli.resume:
                env_cfg.recorders.dataset_export_mode = EnhanceDatasetExportMode.EXPORT_SUCCEEDED_ONLY_RESUME
            else:
                env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
        else:
            if args_cli.resume:
                env_cfg.recorders.dataset_export_mode = EnhanceDatasetExportMode.EXPORT_ALL_RESUME
                assert os.path.exists(
                    args_cli.dataset_file
                ), "The dataset file does not exist. Do not use '--resume' when recording a new dataset."
            else:
                env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
                assert not os.path.exists(
                    args_cli.dataset_file
                ), "The dataset file already exists. Use '--resume' to resume recording."
        env_cfg.recorders.dataset_export_dir_path = output_dir
        env_cfg.recorders.dataset_filename = output_file_name
        if is_direct_env:
            env_cfg.return_success_status = False
        else:
            if not hasattr(env_cfg.terminations, "success"):
                setattr(env_cfg.terminations, "success", None)
            env_cfg.terminations.success = TerminationTermCfg(
                func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            )
    else:
        env_cfg.recorders = None

    # Create environment
    env: ManagerBasedRLEnv | DirectRLEnv = gym.make(task_name, cfg=env_cfg).unwrapped

    # Disable gravity on all robot link prims.
    # The scene uses Left_Robot and Right_Robot prim paths; "Robot" matches both.
    import omni.usd
    from pxr import PhysxSchema, UsdPhysics

    _stage = omni.usd.get_context().get_stage()
    for _prim in _stage.Traverse():
        if "Robot" in str(_prim.GetPath()) and _prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI.Apply(_prim).CreateDisableGravityAttr(True)

    # Manually register a ClothObject on the scene so that the state machine can
    # read cloth particle keypoints via env.scene.particle_objects["cloths"].
    # (The manager-based env does not initialise particle objects automatically;
    # the cloth USD prim is already present in the scene.)
    if not hasattr(env.scene, "particle_objects"):
        env.scene.particle_objects = {}
    env.scene.particle_objects["cloths"] = ClothObject(
        cfg=env.cfg.scene.cloths,
        scene=env.scene,
    )
    env.scene.particle_objects["cloths"].initialize()

    if args_cli.record:
        del env.recorder_manager
        if args_cli.use_lerobot_recorder:
            from leisaac.enhance.datasets.lerobot_dataset_handler import (
                LeRobotDatasetCfg,
            )
            from leisaac.enhance.managers.lerobot_recorder_manager import (
                LeRobotRecorderManager,
            )

            dataset_cfg = LeRobotDatasetCfg(
                repo_id=args_cli.lerobot_dataset_repo_id,
                fps=args_cli.lerobot_dataset_fps,
            )
            env.recorder_manager = LeRobotRecorderManager(env_cfg.recorders, dataset_cfg, env)
        else:
            env.recorder_manager = StreamingRecorderManager(env_cfg.recorders, env)
            env.recorder_manager.flush_steps = 100
            env.recorder_manager.compression = "lzf"

    # Rate limiter
    rate_limiter = RateLimiter(args_cli.step_hz)

    # Initialize / reset
    if hasattr(env, "initialize"):
        env.initialize()
    env.reset()

    resume_recorded_demo_count = 0
    if args_cli.record and args_cli.resume:
        resume_recorded_demo_count = env.recorder_manager._dataset_file_handler.get_num_episodes()
        print(f"Resuming recording from existing dataset with {resume_recorded_demo_count} demonstrations.")
    current_recorded_demo_count = resume_recorded_demo_count

    # --- FK calibration: drive both arms to rest pose and record EE world positions ---
    # cloth_folded() requires both arms within SO101_FOLLOWER_REST_POSE_RANGE.
    # We find the EE world position for each arm at rest pose so the state machine
    # can drive them there via IK during the return-home phase.
    _left_arm = env.scene["left_arm"]
    _right_arm = env.scene["right_arm"]
    _joint_names = list(_left_arm.data.joint_names)
    _REST_POSE_DEG = {
        "shoulder_pan": 0.0,
        "shoulder_lift": -100.0,
        "elbow_flex": 90.0,
        "wrist_flex": 50.0,
        "wrist_roll": 0.0,
        "gripper": -10.0,
    }
    _rest_joint_pos = torch.zeros(env.num_envs, len(_joint_names), device=env.device)
    for _idx, _name in enumerate(_joint_names):
        if _name in _REST_POSE_DEG:
            _rest_joint_pos[:, _idx] = _REST_POSE_DEG[_name] * torch.pi / 180.0

    # Teleport both arms simultaneously to rest pose, then read EE positions via FK.
    _left_arm.write_joint_state_to_sim(
        position=_rest_joint_pos,
        velocity=torch.zeros_like(_rest_joint_pos),
    )
    _right_arm.write_joint_state_to_sim(
        position=_rest_joint_pos,
        velocity=torch.zeros_like(_rest_joint_pos),
    )
    env.sim.step(render=False)
    env.scene.update(dt=env.physics_dt)
    _rest_ee_left = _left_arm.data.body_pos_w[:, -1, :].clone()
    _rest_ee_right = _right_arm.data.body_pos_w[:, -1, :].clone()
    print(f"[Calibration] Left arm rest pose EE (world):  {_rest_ee_left[0].cpu().numpy()}")
    print(f"[Calibration] Right arm rest pose EE (world): {_rest_ee_right[0].cpu().numpy()}")

    # Reset scene to initial state (joints back to default, cloth to initial pose)
    env.reset()
    # -------------------------------------------------------------------------

    sm = FoldClothStateMachine(rest_ee_left=_rest_ee_left, rest_ee_right=_rest_ee_right)
    sm.reset()
    start_record_state = False

    while simulation_app.is_running() and not simulation_app.is_exiting():
        if env.cfg.dynamic_reset_gripper_effort_limit:
            dynamic_reset_gripper_effort_limit_sim(env, device)

        if sm.is_episode_done:
            # --- Check whether the current episode is considered successful ---
            try:
                print("Completed one cycle. Checking task success status...")
                success_tensor = cloth_folded(
                    env,
                    cloth_cfg=SceneEntityCfg("cloths"),
                    cloth_keypoints_index=[159789, 120788, 115370, 159716, 121443, 112382],
                    distance_threshold=0.20,
                )
                success = bool(success_tensor.all().item())
                print("Task success status:", success)
            except Exception as e:
                print("Task failed due to:", e)
                success = False

            if start_record_state:
                if args_cli.record:
                    print("Stop recording.")
                start_record_state = False

            # Only mark the episode as successful if the task succeeds
            if args_cli.record and success:
                print("✅ Task succeeded. Marking this demonstration as SUCCESS.")
                auto_terminate(env, True)
                print("SUCCESS.")
                current_recorded_demo_count += 1
            else:
                print("❌ Task failed. Marking this demonstration as FAILURE.")
                auto_terminate(env, False)

            if (
                args_cli.record
                and env.recorder_manager.exported_successful_episode_count + resume_recorded_demo_count
                > current_recorded_demo_count
            ):
                current_recorded_demo_count = (
                    env.recorder_manager.exported_successful_episode_count + resume_recorded_demo_count
                )
                print(f"Recorded {current_recorded_demo_count} successful demonstrations.")

            if (
                args_cli.record
                and args_cli.num_demos > 0
                and env.recorder_manager.exported_successful_episode_count + resume_recorded_demo_count
                >= args_cli.num_demos
            ):
                print(f"All {args_cli.num_demos} demonstrations have been recorded. Exiting.")
                break

            # Reset cloth particle state before resetting the scene so that the
            # cloth returns to its initial USD pose for the next episode.
            env.scene.particle_objects["cloths"].reset()
            env.reset()
            sm.reset()

            if args_cli.record and args_cli.num_demos > 0 and current_recorded_demo_count >= args_cli.num_demos:
                print(f"All {args_cli.num_demos} demonstrations have been recorded. Exiting.")
                break
        else:
            if not start_record_state:
                if args_cli.record:
                    print("Start recording.")
                start_record_state = True
            actions = sm.get_action(env)
            env.step(actions)
            sm.advance()

        if rate_limiter:
            rate_limiter.sleep(env)

    if args_cli.record and hasattr(env.recorder_manager, "finalize"):
        env.recorder_manager.finalize()

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    # run the main function
    main()
