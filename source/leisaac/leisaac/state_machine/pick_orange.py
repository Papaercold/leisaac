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
_WARMUP_STEPS: int = 30  # physics-settle steps at episode start (first orange only)


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

    MAX_STEPS_PER_ORANGE: int = 420

    def __init__(self, num_oranges: int = 3) -> None:
        self._num_oranges = num_oranges
        self._step_count: int = 0
        self._orange_now: int = 1
        self._episode_done: bool = False

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

        # --- Phase dispatch ---
        if self._orange_now == 1 and step < _WARMUP_STEPS:
            ee_pos_w = env.scene["ee_frame"].data.target_pos_w[:, 0, :].clone()
            target_pos_w, gripper_cmd = self._phase_warmup(ee_pos_w, num_envs, device)
        elif step < 120:
            target_pos_w, gripper_cmd = self._phase_hover_above_orange(orange_pos_w, num_envs, device)
        elif step < 150:
            target_pos_w, gripper_cmd = self._phase_lower_to_orange(orange_pos_w, num_envs, device)
        elif step < 180:
            target_pos_w, gripper_cmd = self._phase_grasp(orange_pos_w, num_envs, device)
        elif step < 220:
            target_pos_w, gripper_cmd = self._phase_lift_orange(orange_pos_w, num_envs, device)
        elif step < 320:
            target_pos_w, gripper_cmd = self._phase_move_above_plate(plate_pos_w, num_envs, device)
        elif step < 350:
            target_pos_w, gripper_cmd = self._phase_lower_to_plate(plate_pos_w, num_envs, device)
        elif step < 380:
            target_pos_w, gripper_cmd = self._phase_release(plate_pos_w, num_envs, device)
        else:
            target_pos_w, gripper_cmd = self._phase_lift_gripper(plate_pos_w, num_envs, device)

        diff_w = target_pos_w - robot_base_pos_w
        target_pos_local = quat_apply(quat_inv(robot_base_quat_w), diff_w)

        return torch.cat([target_pos_local, target_quat, gripper_cmd], dim=-1)

    # ------------------------------------------------------------------
    # Phase methods  (steps 0-419, one method per phase)
    # ------------------------------------------------------------------

    def _phase_warmup(
        self, ee_pos_w: torch.Tensor, num_envs: int, device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Steps 0–_WARMUP_STEPS-1 of the first orange: hold the current EE position.

        Keeps the IK target at the robot's actual end-effector position so the
        physics simulation can settle before any deliberate motion begins.  Only
        active during the first orange's cycle; subsequent oranges skip directly
        to the hover phase.

        Args:
            ee_pos_w: Current end-effector position in world frame, shape ``(num_envs, 3)``.
                Obtained from ``env.scene["ee_frame"].data.target_pos_w[:, 0, :]``.
            num_envs: Number of parallel environments.
            device: Torch device string.

        Returns:
            ``(target_pos_w, gripper_cmd)`` – current EE world position and open gripper command.
        """
        gripper_cmd = torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)
        return ee_pos_w, gripper_cmd

    def _phase_hover_above_orange(
        self, orange_pos_w: torch.Tensor, num_envs: int, device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Steps 0–119: move the gripper to a hover position above the orange.

        Args:
            orange_pos_w: Orange position in world frame, shape ``(num_envs, 3)``.
            num_envs: Number of parallel environments.
            device: Torch device string.

        Returns:
            ``(target_pos_w, gripper_cmd)`` – target world position and gripper command.
        """
        target_pos_w = orange_pos_w.clone()
        target_pos_w[:, 0] -= 0.03
        target_pos_w[:, 2] += 0.1 + _GRIPPER_OFFSET
        gripper_cmd = torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)
        return target_pos_w, gripper_cmd

    def _phase_lower_to_orange(
        self, orange_pos_w: torch.Tensor, num_envs: int, device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Steps 120–149: lower the gripper down to grasp height over the orange.

        Args:
            orange_pos_w: Orange position in world frame, shape ``(num_envs, 3)``.
            num_envs: Number of parallel environments.
            device: Torch device string.

        Returns:
            ``(target_pos_w, gripper_cmd)`` – target world position and gripper command.
        """
        target_pos_w = orange_pos_w.clone()
        target_pos_w[:, 0] -= 0.03
        target_pos_w[:, 2] += _GRIPPER_OFFSET
        gripper_cmd = torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)
        return target_pos_w, gripper_cmd

    def _phase_grasp(
        self, orange_pos_w: torch.Tensor, num_envs: int, device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Steps 150–179: close the gripper to grasp the orange.

        Args:
            orange_pos_w: Orange position in world frame, shape ``(num_envs, 3)``.
            num_envs: Number of parallel environments.
            device: Torch device string.

        Returns:
            ``(target_pos_w, gripper_cmd)`` – target world position and gripper command.
        """
        target_pos_w = orange_pos_w.clone()
        target_pos_w[:, 0] -= 0.03
        target_pos_w[:, 2] += _GRIPPER_OFFSET
        gripper_cmd = torch.full((num_envs, 1), _GRIPPER_CLOSE, device=device)
        return target_pos_w, gripper_cmd

    def _phase_lift_orange(
        self, orange_pos_w: torch.Tensor, num_envs: int, device: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Steps 180–219: lift the grasped orange upward.

        Args:
            orange_pos_w: Orange position in world frame, shape ``(num_envs, 3)``.
            num_envs: Number of parallel environments.
            device: Torch device string.

        Returns:
            ``(target_pos_w, gripper_cmd)`` – target world position and gripper command.
        """
        target_pos_w = orange_pos_w.clone()
        target_pos_w[:, 0] -= 0.03
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
