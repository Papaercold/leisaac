"""State machine for the pick-orange task."""

import math

import torch
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_inv, quat_mul

from .base import StateMachineBase

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_GRIPPER_OPEN = 1.0
_GRIPPER_CLOSE = -1.0
_GRIPPER_OFFSET = 0.1  # vertical clearance for the gripper tip
_APPROACH_STEPS: int = 120  # steps to smoothly interpolate from init EE pos to hover (first orange only)


def _apply_triangle_offset(pos_tensor: torch.Tensor, orange_now: int, radius: float = 0.1) -> torch.Tensor:
    """Apply an equilateral-triangle offset on the x-y plane.

    Distributes up to three target positions evenly around a circle so that
    oranges placed on the plate do not overlap.

    Args:
        pos_tensor: Position tensor of shape ``(num_envs, 3)`` in world coordinates.
            Modified **in-place**.
        orange_now: 1-based index of the current orange (1, 2, or 3).
        radius: Distance from the plate centre to each triangle vertex in metres.

    Returns:
        The modified ``pos_tensor`` (same object, for convenience).
    """
    idx = (orange_now - 1) % 3
    angle = idx * (2 * math.pi / 3)
    pos_tensor[:, 0] += radius * math.cos(angle)
    pos_tensor[:, 1] += radius * math.sin(angle)
    return pos_tensor


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class PickOrangeStateMachine(StateMachineBase):
    """State machine for the pick-orange manipulation task.

    The robot cycles through *num_oranges* oranges.  For each orange it
    executes a fixed sequence of 420 steps that moves the gripper above the
    orange, grasps it, lifts it, transports it to the plate and places it at
    one vertex of an equilateral triangle arrangement.

    Args:
        num_oranges: Total number of oranges to pick and place. Defaults to 3.

    Attributes:
        MAX_STEPS_PER_ORANGE (int): Number of simulation steps per orange cycle.

    Example::

        sm = PickOrangeStateMachine(num_oranges=3)
        env.reset()
        while not sm.is_episode_done:
            actions = sm.get_action(env)
            env.step(actions)
            sm.advance()
        success = task_done(env, ...)
        sm.reset()
    """

    MAX_STEPS_PER_ORANGE: int = 980

    def __init__(self, num_oranges: int = 3, rest_ee_pos_world: torch.Tensor | None = None) -> None:
        self._num_oranges = num_oranges
        self._step_count: int = 0
        self._orange_now: int = 1
        self._episode_done: bool = False
        self._initial_ee_pos: torch.Tensor | None = None
        self._rest_ee_pos_world = rest_ee_pos_world  # (num_envs, 3) world frame, computed by FK calibration

    # ------------------------------------------------------------------
    # StateMachineBase interface
    # ------------------------------------------------------------------

    def get_action(self, env) -> torch.Tensor:
        """Compute the action tensor for the current step.

        Reads object poses from ``env.scene``, dispatches to the appropriate
        phase method, then converts the world-frame target position to the
        robot base frame and assembles the full action tensor.

        The scene is expected to contain:
        - ``"Orange00{orange_now}"`` – the current orange rigid object
        - ``"Plate"`` – the target plate rigid object
        - ``"robot"`` – the manipulator

        Args:
            env: The simulation environment.  Must expose ``env.device``,
                ``env.num_envs``, and ``env.scene``.

        Returns:
            Action tensor of shape ``(num_envs, 8)`` with layout
            ``[pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w, gripper]``
            expressed in the robot base frame.
        """
        robot = env.scene["robot"]
        robot.write_joint_damping_to_sim(damping=10.0)

        device = env.device
        num_envs = env.num_envs
        step = self._step_count

        orange_pos_w = env.scene[f"Orange00{self._orange_now}"].data.root_pos_w.clone()
        plate_pos_w = env.scene["Plate"].data.root_pos_w.clone()
        robot_base_pos_w = env.scene["robot"].data.root_pos_w.clone()
        robot_base_quat_w = env.scene["robot"].data.root_quat_w.clone()

        # Fixed end-effector orientation (no pitch/roll)
        target_quat_w = quat_from_euler_xyz(
            torch.tensor(0.0, device=device),
            torch.tensor(0.0, device=device),
            torch.tensor(0.0, device=device),
        ).repeat(num_envs, 1)
        target_quat = quat_mul(quat_inv(robot_base_quat_w), target_quat_w)

        # Capture initial EE position from robot body data at episode start (step 0, orange 1).
        # body_pos_w is always valid after env.reset() and does not suffer from stale sensor data.
        if self._orange_now == 1 and step == 0:
            self._initial_ee_pos = env.scene["robot"].data.body_pos_w[:, -1, :].clone()
            # 或者直接设置 drive params（依据你 robot API）
        # --- Phase dispatch ---
        if self._orange_now == 1 and step < _APPROACH_STEPS:
            target_pos_w, gripper_cmd = self._phase_approach_hover(orange_pos_w, num_envs, device)
        elif step < 180:
            target_pos_w, gripper_cmd = self._phase_move_above_orange(orange_pos_w, num_envs, device)
        elif step < 300:
            target_pos_w, gripper_cmd = self._phase_hover_above_orange(orange_pos_w, num_envs, device)
        elif step < 360:
            target_pos_w, gripper_cmd = self._phase_lower_to_orange(orange_pos_w, num_envs, device)
        elif step < 420:
            target_pos_w, gripper_cmd = self._phase_grasp(orange_pos_w, num_envs, device)
        elif step < 500:
            target_pos_w, gripper_cmd = self._phase_lift_orange(orange_pos_w, num_envs, device)
        elif step < 550:
            target_pos_w, gripper_cmd = self._phase_move_above_plate(plate_pos_w, num_envs, device)
        elif step < 600:
            target_pos_w, gripper_cmd = self._phase_lower_to_plate(plate_pos_w, num_envs, device)
        elif step < 640:
            target_pos_w, gripper_cmd = self._phase_release(plate_pos_w, num_envs, device)
        elif step < 680:
            target_pos_w, gripper_cmd = self._phase_lift_gripper(plate_pos_w, num_envs, device)
        else:
            target_pos_w, gripper_cmd = self._phase_return_home(num_envs, device)

        diff_w = target_pos_w - robot_base_pos_w
        target_pos_local = quat_apply(quat_inv(robot_base_quat_w), diff_w)

        return torch.cat([target_pos_local, target_quat, gripper_cmd], dim=-1)

    # ------------------------------------------------------------------
    # Phase methods  (steps 0-419, one method per phase)
    # ------------------------------------------------------------------

    def _phase_approach_hover(
        self, orange_pos_w: torch.Tensor, num_envs: int, device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Steps 0–_APPROACH_STEPS-1 of the first orange: smoothly approach hover position.

        Linearly interpolates the IK target from the robot's initial end-effector
        position (captured at step 0 from ``robot.data.body_pos_w``) to the hover
        position above the orange.  This avoids the sudden large IK error that
        occurs when jumping directly from the zero-joint configuration to the hover
        target, which caused the arm to visually drop and oscillate.

        Only active for the first orange; subsequent oranges proceed directly to the
        hover phase because the arm is already in a reasonable working configuration.

        Args:
            orange_pos_w: Orange position in world frame, shape ``(num_envs, 3)``.
            num_envs: Number of parallel environments.
            device: Torch device string.

        Returns:
            ``(target_pos_w, gripper_cmd)`` – interpolated world position and open gripper command.
        """
        hover_target = orange_pos_w.clone()
        hover_target[:, 0] -= 0.03
        hover_target[:, 1] -= 0.01
        hover_target[:, 2] += 0.1 + _GRIPPER_OFFSET

        alpha = self._step_count / _APPROACH_STEPS  # 0.0 at step 0, 1.0 at step _APPROACH_STEPS
        if self._initial_ee_pos is not None:
            target_pos_w = (1.0 - alpha) * self._initial_ee_pos + alpha * hover_target
        else:
            target_pos_w = hover_target

        gripper_cmd = torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)
        return target_pos_w, gripper_cmd

    def _phase_move_above_orange(
        self, orange_pos_w: torch.Tensor, num_envs: int, device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Steps 120–179 (orange 1) / 0–179 (oranges 2-3): move to a high position directly above the orange.

        This phase brings the gripper to a clear altitude above the orange before
        the finer hover and lower phases.  It is the first phase executed for
        oranges 2 and 3 (where the robot transits from the plate side), and runs
        immediately after ``_phase_approach_hover`` for the first orange.

        Args:
            orange_pos_w: Orange position in world frame, shape ``(num_envs, 3)``.
            num_envs: Number of parallel environments.
            device: Torch device string.

        Returns:
            ``(target_pos_w, gripper_cmd)`` – target world position and open gripper command.
        """
        target_pos_w = orange_pos_w.clone()
        target_pos_w[:, 0] -= 0.03
        target_pos_w[:, 1] -= 0.01
        target_pos_w[:, 2] += 0.15 + _GRIPPER_OFFSET
        gripper_cmd = torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)
        return target_pos_w, gripper_cmd

    def _phase_hover_above_orange(
        self, orange_pos_w: torch.Tensor, num_envs: int, device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Steps 180–299: hold the gripper at a hover position above the orange.

        Args:
            orange_pos_w: Orange position in world frame, shape ``(num_envs, 3)``.
            num_envs: Number of parallel environments.
            device: Torch device string.

        Returns:
            ``(target_pos_w, gripper_cmd)`` – target world position and gripper command.
        """
        target_pos_w = orange_pos_w.clone()
        target_pos_w[:, 0] -= 0.03
        target_pos_w[:, 1] -= 0.01
        target_pos_w[:, 2] += 0.1 + _GRIPPER_OFFSET
        gripper_cmd = torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)
        return target_pos_w, gripper_cmd

    def _phase_lower_to_orange(
        self, orange_pos_w: torch.Tensor, num_envs: int, device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Steps 300–359: lower the gripper down to grasp height over the orange.

        Args:
            orange_pos_w: Orange position in world frame, shape ``(num_envs, 3)``.
            num_envs: Number of parallel environments.
            device: Torch device string.

        Returns:
            ``(target_pos_w, gripper_cmd)`` – target world position and gripper command.
        """
        target_pos_w = orange_pos_w.clone()
        target_pos_w[:, 0] -= 0.03
        target_pos_w[:, 1] -= 0.01
        target_pos_w[:, 2] += _GRIPPER_OFFSET
        gripper_cmd = torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)
        return target_pos_w, gripper_cmd

    def _phase_grasp(
        self, orange_pos_w: torch.Tensor, num_envs: int, device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Steps 360–419: close the gripper to grasp the orange.

        Args:
            orange_pos_w: Orange position in world frame, shape ``(num_envs, 3)``.
            num_envs: Number of parallel environments.
            device: Torch device string.

        Returns:
            ``(target_pos_w, gripper_cmd)`` – target world position and gripper command.
        """
        target_pos_w = orange_pos_w.clone()
        target_pos_w[:, 0] -= 0.03
        target_pos_w[:, 1] -= 0.01
        target_pos_w[:, 2] += _GRIPPER_OFFSET
        gripper_cmd = torch.full((num_envs, 1), _GRIPPER_CLOSE, device=device)
        return target_pos_w, gripper_cmd

    def _phase_lift_orange(
        self, orange_pos_w: torch.Tensor, num_envs: int, device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Steps 420–499: lift the grasped orange upward.

        Args:
            orange_pos_w: Orange position in world frame, shape ``(num_envs, 3)``.
            num_envs: Number of parallel environments.
            device: Torch device string.

        Returns:
            ``(target_pos_w, gripper_cmd)`` – target world position and gripper command.
        """
        target_pos_w = orange_pos_w.clone()
        target_pos_w[:, 0] -= 0.03
        target_pos_w[:, 1] -= 0.01
        target_pos_w[:, 2] += 0.25
        gripper_cmd = torch.full((num_envs, 1), _GRIPPER_CLOSE, device=device)
        return target_pos_w, gripper_cmd

    def _phase_move_above_plate(
        self, plate_pos_w: torch.Tensor, num_envs: int, device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Steps 220–319: transport the orange to a hover position above the plate.

        Args:
            plate_pos_w: Plate position in world frame, shape ``(num_envs, 3)``.
            num_envs: Number of parallel environments.
            device: Torch device string.

        Returns:
            ``(target_pos_w, gripper_cmd)`` – target world position and gripper command.
        """
        target_pos_w = plate_pos_w.clone()
        target_pos_w[:, 2] += 0.25
        gripper_cmd = torch.full((num_envs, 1), _GRIPPER_CLOSE, device=device)
        return target_pos_w, gripper_cmd

    def _phase_lower_to_plate(
        self, plate_pos_w: torch.Tensor, num_envs: int, device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Steps 320–349: lower the orange to the placement height on the plate.

        The target position is offset to one vertex of an equilateral triangle
        so that successive oranges do not overlap.

        Args:
            plate_pos_w: Plate position in world frame, shape ``(num_envs, 3)``.
            num_envs: Number of parallel environments.
            device: Torch device string.

        Returns:
            ``(target_pos_w, gripper_cmd)`` – target world position and gripper command.
        """
        target_pos_w = plate_pos_w.clone()
        target_pos_w[:, 2] += _GRIPPER_OFFSET + 0.1
        _apply_triangle_offset(target_pos_w, self._orange_now)
        gripper_cmd = torch.full((num_envs, 1), _GRIPPER_CLOSE, device=device)
        return target_pos_w, gripper_cmd

    def _phase_release(
        self, plate_pos_w: torch.Tensor, num_envs: int, device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Steps 350–379: open the gripper to release the orange onto the plate.

        Args:
            plate_pos_w: Plate position in world frame, shape ``(num_envs, 3)``.
            num_envs: Number of parallel environments.
            device: Torch device string.

        Returns:
            ``(target_pos_w, gripper_cmd)`` – target world position and gripper command.
        """
        target_pos_w = plate_pos_w.clone()
        target_pos_w[:, 2] += _GRIPPER_OFFSET + 0.1
        _apply_triangle_offset(target_pos_w, self._orange_now)
        gripper_cmd = torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)
        return target_pos_w, gripper_cmd

    def _phase_lift_gripper(
        self, plate_pos_w: torch.Tensor, num_envs: int, device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Steps 380–419: lift the gripper clear of the plate after releasing.

        Args:
            plate_pos_w: Plate position in world frame, shape ``(num_envs, 3)``.
            num_envs: Number of parallel environments.
            device: Torch device string.

        Returns:
            ``(target_pos_w, gripper_cmd)`` – target world position and gripper command.
        """
        target_pos_w = plate_pos_w.clone()
        target_pos_w[:, 2] += 0.2
        _apply_triangle_offset(target_pos_w, self._orange_now)
        gripper_cmd = torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)
        return target_pos_w, gripper_cmd

    def _phase_return_home(
        self, num_envs: int, device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Steps 620–919: return the gripper to the robot's rest pose end-effector position.

        Targets the EE position that was computed by FK calibration at startup
        (``rest_ee_pos_world``), which corresponds to the SO-101 rest pose joints
        (shoulder_lift ≈ -100°, elbow_flex ≈ 90°, wrist_flex ≈ 50°).  The
        incremental IK will drive the arm toward this position over the 300 steps
        (~5 s at 60 Hz), giving it enough time to converge within the ±30° joint
        tolerance required by ``task_done``.

        Falls back to ``_initial_ee_pos`` (zero-joint EE) if calibration was
        not performed.

        Args:
            num_envs: Number of parallel environments.
            device: Torch device string.

        Returns:
            ``(target_pos_w, gripper_cmd)`` – rest-pose world position and open gripper command.
        """
        if self._rest_ee_pos_world is not None:
            target_pos_w = self._rest_ee_pos_world.clone()
        elif self._initial_ee_pos is not None:
            target_pos_w = self._initial_ee_pos.clone()
        else:
            target_pos_w = torch.zeros(num_envs, 3, device=device)
        gripper_cmd = torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)
        return target_pos_w, gripper_cmd

    def advance(self) -> None:
        """Advance the internal step counter and manage orange transitions.

        When the current orange's cycle completes (``step_count`` reaches
        :attr:`MAX_STEPS_PER_ORANGE`), the machine either:

        - increments ``orange_now`` and resets ``step_count`` if more oranges
          remain, or
        - sets :attr:`is_episode_done` to ``True`` if all oranges are done.
        """
        self._step_count += 1
        if self._step_count >= self.MAX_STEPS_PER_ORANGE:
            if self._orange_now >= self._num_oranges:
                self._episode_done = True
            else:
                self._orange_now += 1
                self._step_count = 0

    def reset(self) -> None:
        """Reset the state machine to its initial state for a new episode."""
        self._step_count = 0
        self._orange_now = 1
        self._episode_done = False
        self._initial_ee_pos = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_episode_done(self) -> bool:
        """``True`` once all oranges have been picked and placed."""
        return self._episode_done

    @property
    def orange_now(self) -> int:
        """1-based index of the orange currently being handled."""
        return self._orange_now

    @property
    def step_count(self) -> int:
        """Number of steps elapsed within the current orange's cycle."""
        return self._step_count
