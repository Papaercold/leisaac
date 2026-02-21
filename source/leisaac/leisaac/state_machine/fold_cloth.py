"""State machine for the fold-cloth task (bi-arm SO-101)."""

import torch
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_inv, quat_mul

from .base import StateMachineBase

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_GRIPPER_OPEN = 1.0
_GRIPPER_CLOSE = -1.0

# Cloth particle indices for the 6 keypoints, matching cloth_folded() in terminations.py.
# Order: left_sleeve(0), left_shoulder(1), left_hem(2),
#        right_sleeve(3), right_shoulder(4), right_hem(5)
_CLOTH_KEYPOINTS = [159789, 120788, 115370, 159716, 121443, 112382]

_KP_LEFT_SLEEVE = 0
_KP_LEFT_SHOULDER = 1
_KP_LEFT_HEM = 2
_KP_RIGHT_SLEEVE = 3
_KP_RIGHT_SHOULDER = 4
_KP_RIGHT_HEM = 5

# Height offsets relative to cloth keypoints
_HOVER_ABOVE = 0.12   # approach height above keypoint (m)
_GRASP_ABOVE = 0.02   # grip height above cloth surface (cloth is thin) (m)
_FOLD_HEIGHT = 0.18   # height above shoulder target while carrying cloth (m)
_LIFT_HEIGHT = 0.20   # neutral lift height used between phases (m)

# ---------------------------------------------------------------------------
# Phase step boundaries (at 60 Hz)
# ---------------------------------------------------------------------------
#   0 – 119  hover_above_sleeves       (2 s) – both arms approach above sleeves
# 120 – 179  lower_to_sleeves          (1 s) – descend to grasp height
# 180 – 239  grasp_sleeves             (1 s) – close grippers
# 240 – 359  cross_sleeves             (2 s) – left→right_shoulder, right→left_shoulder
# 360 – 419  release_sleeves           (1 s) – open grippers at crossed position
# 420 – 479  lift_grippers_1           (1 s) – lift clear of cloth
# 480 – 599  hover_above_hems          (2 s) – move above bottom corners
# 600 – 659  lower_to_hems             (1 s) – descend to grasp height
# 660 – 719  grasp_hems                (1 s) – close grippers
# 720 – 839  fold_hems_up              (2 s) – left_hem→left_shoulder, right_hem→right_shoulder
# 840 – 899  release_hems              (1 s) – open grippers
# 900 – 959  lift_grippers_2           (1 s) – lift clear of cloth
# 960 –1259  return_home               (5 s) – both arms drive to rest-pose EE
_P_HOVER_SLEEVES = 120
_P_LOWER_SLEEVES = 180
_P_GRASP_SLEEVES = 240
_P_CROSS_SLEEVES = 360
_P_RELEASE_SLEEVES = 420
_P_LIFT1 = 480
_P_HOVER_HEMS = 600
_P_LOWER_HEMS = 660
_P_GRASP_HEMS = 720
_P_FOLD_HEMS = 840
_P_RELEASE_HEMS = 900
_P_LIFT2 = 960


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class FoldClothStateMachine(StateMachineBase):
    """Bi-arm state machine for the fold-cloth manipulation task.

    The robot uses two SO-101 arms to fold a garment lying on a surface:

    1. Both arms reach above the sleeves, grasp them and cross them over to
       opposite shoulders (left sleeve → right shoulder, right sleeve → left
       shoulder).
    2. Both arms then reach the hem corners, grasp them and fold them upward
       to the corresponding shoulder positions.
    3. Both arms return to their rest poses so that ``cloth_folded()``
       (which also checks ``is_so101_at_rest_pose`` for both arms) can
       evaluate the episode as successful.

    Args:
        rest_ee_left:  World-frame EE position of the left arm at rest pose,
            shape ``(num_envs, 3)``.  Computed by FK calibration at startup.
        rest_ee_right: World-frame EE position of the right arm at rest pose,
            shape ``(num_envs, 3)``.  Computed by FK calibration at startup.

    Attributes:
        MAX_STEPS (int): Total simulation steps for one episode (1260 ≈ 21 s at 60 Hz).

    Example::

        sm = FoldClothStateMachine(rest_ee_left=left_ee, rest_ee_right=right_ee)
        env.reset()
        while not sm.is_episode_done:
            actions = sm.get_action(env)
            env.step(actions)
            sm.advance()
        success = cloth_folded(env, ...)
        sm.reset()
    """

    MAX_STEPS: int = 1260

    def __init__(
        self,
        rest_ee_left: torch.Tensor | None = None,
        rest_ee_right: torch.Tensor | None = None,
    ) -> None:
        self._step_count: int = 0
        self._episode_done: bool = False
        self._rest_ee_left = rest_ee_left    # (num_envs, 3), calibrated at startup
        self._rest_ee_right = rest_ee_right  # (num_envs, 3), calibrated at startup
        # Shoulder positions captured at step 0 and used as stable fold targets.
        self._left_shoulder_pos: torch.Tensor | None = None
        self._right_shoulder_pos: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # StateMachineBase interface
    # ------------------------------------------------------------------

    def get_action(self, env) -> torch.Tensor:
        """Compute the 16-D action tensor for both arms at the current step.

        The action layout is:
        ``[left_pos(3), left_quat(4), left_gripper(1), right_pos(3), right_quat(4), right_gripper(1)]``

        The scene must expose:
        - ``"left_arm"`` / ``"right_arm"`` – SO-101 articulations
        - ``env.scene.particle_objects["cloths"]`` – :class:`ClothObject`

        Args:
            env: The simulation environment.

        Returns:
            Action tensor of shape ``(num_envs, 16)``.
        """
        step = self._step_count
        num_envs = env.num_envs
        device = env.device

        # Read cloth keypoints: (num_envs, 6, 3) in world frame
        kp = self._get_cloth_keypoints(env)

        # Capture shoulder positions once at episode start as stable fold targets
        if step == 0:
            self._left_shoulder_pos = kp[:, _KP_LEFT_SHOULDER, :].clone()
            self._right_shoulder_pos = kp[:, _KP_RIGHT_SHOULDER, :].clone()

        # --- Phase dispatch ---
        if step < _P_HOVER_SLEEVES:
            l_tgt, r_tgt, l_grip, r_grip = self._phase_hover_sleeves(kp, num_envs, device)
        elif step < _P_LOWER_SLEEVES:
            l_tgt, r_tgt, l_grip, r_grip = self._phase_lower_to_sleeves(kp, num_envs, device)
        elif step < _P_GRASP_SLEEVES:
            l_tgt, r_tgt, l_grip, r_grip = self._phase_grasp_sleeves(kp, num_envs, device)
        elif step < _P_CROSS_SLEEVES:
            l_tgt, r_tgt, l_grip, r_grip = self._phase_cross_sleeves(num_envs, device)
        elif step < _P_RELEASE_SLEEVES:
            l_tgt, r_tgt, l_grip, r_grip = self._phase_release_sleeves(num_envs, device)
        elif step < _P_LIFT1:
            l_tgt, r_tgt, l_grip, r_grip = self._phase_lift_grippers(num_envs, device)
        elif step < _P_HOVER_HEMS:
            l_tgt, r_tgt, l_grip, r_grip = self._phase_hover_hems(kp, num_envs, device)
        elif step < _P_LOWER_HEMS:
            l_tgt, r_tgt, l_grip, r_grip = self._phase_lower_to_hems(kp, num_envs, device)
        elif step < _P_GRASP_HEMS:
            l_tgt, r_tgt, l_grip, r_grip = self._phase_grasp_hems(kp, num_envs, device)
        elif step < _P_FOLD_HEMS:
            l_tgt, r_tgt, l_grip, r_grip = self._phase_fold_hems(num_envs, device)
        elif step < _P_RELEASE_HEMS:
            l_tgt, r_tgt, l_grip, r_grip = self._phase_release_hems(num_envs, device)
        elif step < _P_LIFT2:
            l_tgt, r_tgt, l_grip, r_grip = self._phase_lift_grippers(num_envs, device)
        else:
            l_tgt, r_tgt, l_grip, r_grip = self._phase_return_home(num_envs, device)

        left_action = self._to_arm_action(l_tgt, l_grip, "left_arm", env)
        right_action = self._to_arm_action(r_tgt, r_grip, "right_arm", env)

        return torch.cat([left_action, right_action], dim=-1)  # (num_envs, 16)

    def advance(self) -> None:
        """Increment the step counter; mark episode done after MAX_STEPS."""
        self._step_count += 1
        if self._step_count >= self.MAX_STEPS:
            self._episode_done = True

    def reset(self) -> None:
        """Reset to the initial state for a new episode."""
        self._step_count = 0
        self._episode_done = False
        self._left_shoulder_pos = None
        self._right_shoulder_pos = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_cloth_keypoints(self, env) -> torch.Tensor:
        """Return cloth keypoints as ``(num_envs, 6, 3)`` in world frame."""
        cloth = env.scene.particle_objects["cloths"]
        return cloth.point_positions[:, _CLOTH_KEYPOINTS, :]

    def _to_arm_action(
        self,
        target_pos_w: torch.Tensor,
        gripper_cmd: torch.Tensor,
        arm_name: str,
        env,
    ) -> torch.Tensor:
        """Convert a world-frame EE position to the arm's local IK action (8-D).

        Args:
            target_pos_w: Target position in world frame, shape ``(num_envs, 3)``.
            gripper_cmd:  Gripper command, shape ``(num_envs, 1)``.
            arm_name:     ``"left_arm"`` or ``"right_arm"``.
            env:          The simulation environment.

        Returns:
            Tensor of shape ``(num_envs, 8)`` = ``[pos(3), quat(4), gripper(1)]``
            expressed in the arm's base frame.
        """
        device = env.device
        num_envs = env.num_envs

        base_pos_w = env.scene[arm_name].data.root_pos_w.clone()
        base_quat_w = env.scene[arm_name].data.root_quat_w.clone()

        # Identity EE orientation in world frame (gripper pointing down)
        target_quat_w = quat_from_euler_xyz(
            torch.tensor(0.0, device=device),
            torch.tensor(0.0, device=device),
            torch.tensor(0.0, device=device),
        ).repeat(num_envs, 1)
        target_quat = quat_mul(quat_inv(base_quat_w), target_quat_w)

        diff_w = target_pos_w - base_pos_w
        target_pos_local = quat_apply(quat_inv(base_quat_w), diff_w)

        return torch.cat([target_pos_local, target_quat, gripper_cmd], dim=-1)

    def _shoulder_center(self, num_envs: int, device: str) -> torch.Tensor:
        """Mid-point between the two captured shoulder positions."""
        if self._left_shoulder_pos is not None and self._right_shoulder_pos is not None:
            return (self._left_shoulder_pos + self._right_shoulder_pos) / 2.0
        return torch.zeros(num_envs, 3, device=device)

    # ------------------------------------------------------------------
    # Phase methods
    # ------------------------------------------------------------------

    def _phase_hover_sleeves(
        self, kp: torch.Tensor, num_envs: int, device: str
    ) -> tuple:
        """Steps 0–119: both arms hover above respective sleeves."""
        l_tgt = kp[:, _KP_LEFT_SLEEVE, :].clone()
        l_tgt[:, 2] += _HOVER_ABOVE
        r_tgt = kp[:, _KP_RIGHT_SLEEVE, :].clone()
        r_tgt[:, 2] += _HOVER_ABOVE
        open_g = torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)
        return l_tgt, r_tgt, open_g, open_g.clone()

    def _phase_lower_to_sleeves(
        self, kp: torch.Tensor, num_envs: int, device: str
    ) -> tuple:
        """Steps 120–179: descend to grasp height above sleeves."""
        l_tgt = kp[:, _KP_LEFT_SLEEVE, :].clone()
        l_tgt[:, 2] += _GRASP_ABOVE
        r_tgt = kp[:, _KP_RIGHT_SLEEVE, :].clone()
        r_tgt[:, 2] += _GRASP_ABOVE
        open_g = torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)
        return l_tgt, r_tgt, open_g, open_g.clone()

    def _phase_grasp_sleeves(
        self, kp: torch.Tensor, num_envs: int, device: str
    ) -> tuple:
        """Steps 180–239: close grippers on sleeves."""
        l_tgt = kp[:, _KP_LEFT_SLEEVE, :].clone()
        l_tgt[:, 2] += _GRASP_ABOVE
        r_tgt = kp[:, _KP_RIGHT_SLEEVE, :].clone()
        r_tgt[:, 2] += _GRASP_ABOVE
        close_g = torch.full((num_envs, 1), _GRIPPER_CLOSE, device=device)
        return l_tgt, r_tgt, close_g, close_g.clone()

    def _phase_cross_sleeves(self, num_envs: int, device: str) -> tuple:
        """Steps 240–359: cross sleeves to opposite shoulders.

        Left arm (holding left sleeve) moves to right shoulder.
        Right arm (holding right sleeve) moves to left shoulder.
        """
        if self._right_shoulder_pos is not None:
            l_tgt = self._right_shoulder_pos.clone()
            l_tgt[:, 2] += _FOLD_HEIGHT
        else:
            l_tgt = torch.zeros(num_envs, 3, device=device)

        if self._left_shoulder_pos is not None:
            r_tgt = self._left_shoulder_pos.clone()
            r_tgt[:, 2] += _FOLD_HEIGHT
        else:
            r_tgt = torch.zeros(num_envs, 3, device=device)

        close_g = torch.full((num_envs, 1), _GRIPPER_CLOSE, device=device)
        return l_tgt, r_tgt, close_g, close_g.clone()

    def _phase_release_sleeves(self, num_envs: int, device: str) -> tuple:
        """Steps 360–419: open grippers to deposit sleeves at shoulder positions."""
        if self._right_shoulder_pos is not None:
            l_tgt = self._right_shoulder_pos.clone()
            l_tgt[:, 2] += _FOLD_HEIGHT
        else:
            l_tgt = torch.zeros(num_envs, 3, device=device)

        if self._left_shoulder_pos is not None:
            r_tgt = self._left_shoulder_pos.clone()
            r_tgt[:, 2] += _FOLD_HEIGHT
        else:
            r_tgt = torch.zeros(num_envs, 3, device=device)

        open_g = torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)
        return l_tgt, r_tgt, open_g, open_g.clone()

    def _phase_lift_grippers(self, num_envs: int, device: str) -> tuple:
        """Lift both grippers clear of the cloth to a neutral position."""
        center = self._shoulder_center(num_envs, device)
        l_tgt = center.clone()
        l_tgt[:, 2] += _LIFT_HEIGHT + 0.10
        r_tgt = center.clone()
        r_tgt[:, 2] += _LIFT_HEIGHT + 0.10
        open_g = torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)
        return l_tgt, r_tgt, open_g, open_g.clone()

    def _phase_hover_hems(
        self, kp: torch.Tensor, num_envs: int, device: str
    ) -> tuple:
        """Steps 480–599: hover above the bottom hem corners."""
        l_tgt = kp[:, _KP_LEFT_HEM, :].clone()
        l_tgt[:, 2] += _HOVER_ABOVE
        r_tgt = kp[:, _KP_RIGHT_HEM, :].clone()
        r_tgt[:, 2] += _HOVER_ABOVE
        open_g = torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)
        return l_tgt, r_tgt, open_g, open_g.clone()

    def _phase_lower_to_hems(
        self, kp: torch.Tensor, num_envs: int, device: str
    ) -> tuple:
        """Steps 600–659: descend to grasp height above hems."""
        l_tgt = kp[:, _KP_LEFT_HEM, :].clone()
        l_tgt[:, 2] += _GRASP_ABOVE
        r_tgt = kp[:, _KP_RIGHT_HEM, :].clone()
        r_tgt[:, 2] += _GRASP_ABOVE
        open_g = torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)
        return l_tgt, r_tgt, open_g, open_g.clone()

    def _phase_grasp_hems(
        self, kp: torch.Tensor, num_envs: int, device: str
    ) -> tuple:
        """Steps 660–719: close grippers on hem corners."""
        l_tgt = kp[:, _KP_LEFT_HEM, :].clone()
        l_tgt[:, 2] += _GRASP_ABOVE
        r_tgt = kp[:, _KP_RIGHT_HEM, :].clone()
        r_tgt[:, 2] += _GRASP_ABOVE
        close_g = torch.full((num_envs, 1), _GRIPPER_CLOSE, device=device)
        return l_tgt, r_tgt, close_g, close_g.clone()

    def _phase_fold_hems(self, num_envs: int, device: str) -> tuple:
        """Steps 720–839: fold hem corners up to shoulder level.

        Left hem moves to left shoulder position; right hem to right shoulder.
        """
        if self._left_shoulder_pos is not None:
            l_tgt = self._left_shoulder_pos.clone()
            l_tgt[:, 2] += _FOLD_HEIGHT
        else:
            l_tgt = torch.zeros(num_envs, 3, device=device)

        if self._right_shoulder_pos is not None:
            r_tgt = self._right_shoulder_pos.clone()
            r_tgt[:, 2] += _FOLD_HEIGHT
        else:
            r_tgt = torch.zeros(num_envs, 3, device=device)

        close_g = torch.full((num_envs, 1), _GRIPPER_CLOSE, device=device)
        return l_tgt, r_tgt, close_g, close_g.clone()

    def _phase_release_hems(self, num_envs: int, device: str) -> tuple:
        """Steps 840–899: open grippers to deposit hems at shoulder positions."""
        if self._left_shoulder_pos is not None:
            l_tgt = self._left_shoulder_pos.clone()
            l_tgt[:, 2] += _FOLD_HEIGHT
        else:
            l_tgt = torch.zeros(num_envs, 3, device=device)

        if self._right_shoulder_pos is not None:
            r_tgt = self._right_shoulder_pos.clone()
            r_tgt[:, 2] += _FOLD_HEIGHT
        else:
            r_tgt = torch.zeros(num_envs, 3, device=device)

        open_g = torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)
        return l_tgt, r_tgt, open_g, open_g.clone()

    def _phase_return_home(self, num_envs: int, device: str) -> tuple:
        """Steps 960–1259: drive both arms to rest-pose EE (for task_done check).

        Targets ``rest_ee_left`` / ``rest_ee_right`` computed at startup by FK
        calibration.  The incremental IK drives joints toward the rest pose
        configuration (shoulder_lift≈-100°, elbow_flex≈90°, wrist_flex≈50°)
        within the ±30° tolerance required by ``is_so101_at_rest_pose``.
        """
        l_tgt = self._rest_ee_left.clone() if self._rest_ee_left is not None else torch.zeros(num_envs, 3, device=device)
        r_tgt = self._rest_ee_right.clone() if self._rest_ee_right is not None else torch.zeros(num_envs, 3, device=device)
        open_g = torch.full((num_envs, 1), _GRIPPER_OPEN, device=device)
        return l_tgt, r_tgt, open_g, open_g.clone()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_episode_done(self) -> bool:
        """``True`` once all ``MAX_STEPS`` have been executed."""
        return self._episode_done

    @property
    def step_count(self) -> int:
        """Number of steps elapsed in the current episode."""
        return self._step_count
