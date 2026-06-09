import json
from pathlib import Path

import numpy as np

from .math_utils import quat_rotate_inverse_wxyz, yaw_quat


UNITREE_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


class MotionReference:
    def __init__(self, motion_path, future_steps, joint_names, body_names, root_body_name="pelvis"):
        self.motion_path = Path(motion_path)
        with open(self.motion_path / "meta.json", "r") as f:
            meta = json.load(f)
        if float(meta["fps"]) != 50.0:
            raise ValueError(f"Only 50 Hz motion files are supported, got fps={meta['fps']}")

        motion_files = sorted(self.motion_path.glob("*.npz"))
        if len(motion_files) != 1:
            raise ValueError(f"Expected exactly one motion npz in {self.motion_path}, got {len(motion_files)}")
        motion = dict(np.load(motion_files[0]))

        self.body_names = meta["body_names"]
        self.joint_names = self._ordered_joint_names(meta["joint_names"])
        self.body_pos_w = motion["body_pos_w"]
        self.body_quat_w = motion["body_quat_w"]
        self.joint_pos = self._reorder_joints(motion["joint_pos"], meta["joint_names"])
        self.object_geo = motion["object_points"].reshape(-1, 3)

        self.future_steps = np.asarray(future_steps, dtype=np.int64)
        self.joint_indices = [self.joint_names.index(name) for name in joint_names]
        self.body_indices = [self.body_names.index(name) for name in body_names]
        self.root_body_idx = self.body_names.index(root_body_name)
        self.t = 0
        self.motion_steps = self.joint_pos.shape[0]

    @staticmethod
    def _ordered_joint_names(source_joint_names):
        extra_names = [name for name in source_joint_names if name not in UNITREE_JOINT_NAMES]
        return UNITREE_JOINT_NAMES + extra_names

    def _reorder_joints(self, joint_data, source_joint_names):
        out = np.zeros((joint_data.shape[0], len(self.joint_names)), dtype=joint_data.dtype)
        for source_idx, name in enumerate(source_joint_names):
            out[:, self.joint_names.index(name)] = joint_data[:, source_idx]
        return out

    def update(self):
        self.t = (self.t + 1) % self.motion_steps
        future_idx = np.clip(self.t + self.future_steps, 0, self.motion_steps - 1)
        self.ref_joint_pos_future = self.joint_pos[future_idx][:, self.joint_indices]
        self.ref_body_pos_future_w = self.body_pos_w[future_idx][:, self.body_indices]
        self.ref_root_pos_w = self.body_pos_w[self.t, self.root_body_idx]
        self.ref_root_quat_w = self._root_quat_at(self.t)

    def _root_quat_at(self, step):
        return self.body_quat_w[step, self.root_body_idx]

    def get_ref_body_pos_future_local(self):
        root_pos = np.tile(self.ref_root_pos_w, (len(self.future_steps), len(self.body_indices), 1))
        root_quat = np.tile(self.ref_root_quat_w, (len(self.future_steps), len(self.body_indices), 1))
        root_pos[..., 2] = 0.0
        root_quat = yaw_quat(root_quat)
        return quat_rotate_inverse_wxyz(root_quat, self.ref_body_pos_future_w - root_pos).reshape(-1)

    def get_ref_joint_pos_future(self):
        return self.ref_joint_pos_future.reshape(-1)

    def get_ref_motion_phase(self):
        return np.array([self.t / self.motion_steps], dtype=np.float32)

    def get_object_geo(self):
        return self.object_geo.reshape(-1)
