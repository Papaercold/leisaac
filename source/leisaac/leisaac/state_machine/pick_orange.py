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

        Reads object poses from ``env.scene`` and returns a 8-DOF action
        tensor ``[pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w, gripper]``
        expressed in the robot base frame.

        The scene is expected to contain:
        - ``"Orange00{orange_now}"`` – the current orange rigid object
        - ``"Plate"`` – the target plate rigid object
        - ``"robot"`` – the manipulator

        Args:
            env: The simulation environment.  Must expose ``env.device``,
                ``env.num_envs``, and ``env.scene``.

        Returns:
            Action tensor of shape ``(num_envs, 8)``.
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

        gripper_cmd = torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)

        # --- Phase selection based on step count ---
        if step < 120:
            # Move above orange (hover)
            target_pos_w = orange_pos_w.clone()
            target_pos_w[:, 0] -= 0.03
            target_pos_w[:, 2] += 0.1 + _GRIPPER_OFFSET
            gripper_cmd[:] = _GRIPPER_OPEN
        elif step < 150:
            # Lower to orange
            target_pos_w = orange_pos_w.clone()
            target_pos_w[:, 0] -= 0.03
            target_pos_w[:, 2] += _GRIPPER_OFFSET
            gripper_cmd[:] = _GRIPPER_OPEN
        elif step < 180:
            # Close gripper (grasp)
            target_pos_w = orange_pos_w.clone()
            target_pos_w[:, 0] -= 0.03
            target_pos_w[:, 2] += _GRIPPER_OFFSET
            gripper_cmd[:] = _GRIPPER_CLOSE
        elif step < 220:
            # Lift orange
            target_pos_w = orange_pos_w.clone()
            target_pos_w[:, 0] -= 0.03
            target_pos_w[:, 2] += 0.25
            gripper_cmd[:] = _GRIPPER_CLOSE
        elif step < 320:
            # Move above plate
            target_pos_w = plate_pos_w.clone()
            target_pos_w[:, 2] += 0.25
            gripper_cmd[:] = _GRIPPER_CLOSE
        elif step < 350:
            # Lower to plate
            target_pos_w = plate_pos_w.clone()
            target_pos_w[:, 2] += _GRIPPER_OFFSET + 0.1
            _apply_triangle_offset(target_pos_w, self._orange_now)
            gripper_cmd[:] = _GRIPPER_CLOSE
        elif step < 380:
            # Release orange
            target_pos_w = plate_pos_w.clone()
            target_pos_w[:, 2] += _GRIPPER_OFFSET + 0.1
            _apply_triangle_offset(target_pos_w, self._orange_now)
            gripper_cmd[:] = _GRIPPER_OPEN
        else:
            # Lift gripper clear of the plate
            target_pos_w = plate_pos_w.clone()
            target_pos_w[:, 2] += 0.2
            _apply_triangle_offset(target_pos_w, self._orange_now)
            gripper_cmd[:] = _GRIPPER_OPEN

        diff_w = target_pos_w - robot_base_pos_w
        target_pos_local = quat_apply(quat_inv(robot_base_quat_w), diff_w)

        return torch.cat([target_pos_local, target_quat, gripper_cmd], dim=-1)

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
