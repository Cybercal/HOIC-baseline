from dataclasses import dataclass

import numpy as np
import onnxruntime

from .math_utils import quat_rotate_inverse_xyzw
from .motion import MotionReference


@dataclass
class PolicyOutput:
    target_dof_pos: np.ndarray
    dof_kps: np.ndarray
    dof_kds: np.ndarray
    tau_ff: np.ndarray


class InterMimicRod:
    def __init__(self, policy_path, motion_path):
        self.num_actions = 23
        self.num_dof = 29
        self.policy = onnxruntime.InferenceSession(str(policy_path))
        self.policy_inputs = self.policy.get_inputs()
        self.input_names = [input_info.name for input_info in self.policy_inputs]
        self.output_names = [output.name for output in self.policy.get_outputs()]

        self.cor_hdmi_norm = [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 12, 16, 20, 22]
        self.cor_obs_index = [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
        self.history_steps = [0, 1, 2, 3, 4, 8]

        self.refmotion = MotionReference(
            motion_path=motion_path,
            future_steps=[1, 2, 8, 16, 32],
            body_names=[
                "pelvis", "left_hip_pitch_link", "right_hip_pitch_link",
                "left_hip_yaw_link", "right_hip_yaw_link", "torso_link",
                "left_knee_link", "right_knee_link", "left_shoulder_pitch_link",
                "right_shoulder_pitch_link", "left_ankle_roll_link", "right_ankle_roll_link",
                "left_elbow_link", "right_elbow_link", "left_wrist_yaw_link", "right_wrist_yaw_link",
            ],
            joint_names=[
                "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
                "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
                "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
                "left_knee_joint", "right_knee_joint", "left_shoulder_pitch_joint",
                "right_shoulder_pitch_joint", "left_ankle_pitch_joint", "right_ankle_pitch_joint",
                "left_shoulder_roll_joint", "right_shoulder_roll_joint", "left_ankle_roll_joint",
                "right_ankle_roll_joint", "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
                "left_elbow_joint", "right_elbow_joint",
            ],
        )

        self.default_pos = np.array([
            -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
            -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
            0.0, 0.0, 0.0,
            0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
            0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
        ], dtype=np.float32)
        self.action_scale = np.array([
            0.55, 0.35, 0.55, 0.35, 0.44, 0.44,
            0.55, 0.35, 0.55, 0.35, 0.44, 0.44,
            0.55, 0.44, 0.44,
            0.44, 0.44, 0.44, 0.44, 0.0, 0.0, 0.0,
            0.44, 0.44, 0.44, 0.44, 0.0, 0.0, 0.0,
        ], dtype=np.float32)
        self.dof_kps = np.array([
            40.17923847137318, 99.09842777666113, 40.17923847137318, 99.09842777666113, 28.50124619574858, 28.50124619574858,
            40.17923847137318, 99.09842777666113, 40.17923847137318, 99.09842777666113, 28.50124619574858, 28.50124619574858,
            40.17923847137318, 28.50124619574858, 28.50124619574858,
            14.25062309787429, 14.25062309787429, 14.25062309787429, 14.25062309787429, 14.25062309787429, 16.77832748089279, 16.77832748089279,
            14.25062309787429, 14.25062309787429, 14.25062309787429, 14.25062309787429, 14.25062309787429, 16.77832748089279, 16.77832748089279,
        ], dtype=np.float32)
        self.dof_kds = np.array([
            2.5578897650279457, 6.3088018534966395, 2.5578897650279457, 6.3088018534966395, 1.814445686584846, 1.814445686584846,
            2.5578897650279457, 6.3088018534966395, 2.5578897650279457, 6.3088018534966395, 1.814445686584846, 1.814445686584846,
            2.5578897650279457, 1.814445686584846, 1.814445686584846,
            0.907222843292423, 0.907222843292423, 0.907222843292423, 0.907222843292423, 0.907222843292423, 1.06814150219, 1.06814150219,
            0.907222843292423, 0.907222843292423, 0.907222843292423, 0.907222843292423, 0.907222843292423, 1.06814150219, 1.06814150219,
        ], dtype=np.float32)
        self.torque_limits = np.array([
            88.0, 139.0, 88.0, 139.0, 50.0, 50.0,
            88.0, 139.0, 88.0, 139.0, 50.0, 50.0,
            88.0, 50.0, 50.0,
            25.0, 25.0, 25.0, 25.0, 5.0, 5.0, 5.0,
            25.0, 25.0, 25.0, 25.0, 5.0, 5.0, 5.0,
        ], dtype=np.float32)
        self.dof_pos_buf = np.tile(self.default_pos, (9, 1))
        self.action_buf = np.zeros((23, 3), dtype=np.float32)
        self.action = np.zeros(self.num_actions, dtype=np.float32)

    def reset_history(self):
        self.dof_pos_buf[:, :] = 0.0
        self.action_buf[:, :] = 0.0
        self.action[:] = 0.0

    def step(self, state):
        self.refmotion.update()
        obs_dict = self._compute_obs(state)
        policy_feed = self._build_policy_feed(obs_dict)
        outputs = self.policy.run(None, policy_feed)
        action = outputs[self.output_names.index("action")].squeeze(0) if "action" in self.output_names else outputs[10].squeeze(0)
        self.action = action.copy()
        action = action[self.cor_hdmi_norm].copy()
        action = self._add_wrist_6dof(action)
        target_dof_pos = action * self.action_scale + self.default_pos
        return PolicyOutput(
            target_dof_pos=target_dof_pos.astype(np.float32),
            dof_kps=self.dof_kps,
            dof_kds=self.dof_kds,
            tau_ff=np.zeros(self.num_dof, dtype=np.float32),
        )

    def _compute_obs(self, state):
        gravity_vec = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        projected_gravity = quat_rotate_inverse_xyzw(state["quat"], gravity_vec)

        self.dof_pos_buf[1:, :] = self.dof_pos_buf[:-1, :].copy()
        self.action_buf[:, 1:] = self.action_buf[:, :-1].copy()
        self.dof_pos_buf[0, :] = state["qj"][self.cor_obs_index].copy()
        self.action_buf[:, 0] = self.action.copy()

        obs_policy = np.concatenate([
            state["ang_vel"],
            projected_gravity,
            self.dof_pos_buf[self.history_steps].reshape(-1),
            self.action_buf.reshape(-1),
        ], axis=-1)
        obs_command = np.concatenate([
            self.refmotion.get_ref_body_pos_future_local(),
            self.refmotion.get_ref_joint_pos_future(),
            self.refmotion.get_ref_motion_phase(),
        ], axis=-1)
        return {
            "command": obs_command[None, :].astype(np.float32),
            "policy": obs_policy[None, :].astype(np.float32),
            "object_geo": self.refmotion.get_object_geo()[None, :].astype(np.float32),
        }

    def _build_policy_feed(self, obs_dict):
        feed = {}
        INPUT_NAME_MAPPING = {
            "object_geo_": "object_geo", 
        }
        for input_info in self.policy_inputs:
            if input_info.name not in obs_dict:
                if input_info.name in INPUT_NAME_MAPPING and INPUT_NAME_MAPPING[input_info.name] in obs_dict:
                    feed[input_info.name] = self._match_input_shape(obs_dict[INPUT_NAME_MAPPING[input_info.name]], input_info)
                    continue
                raise KeyError(f"Policy expects input '{input_info.name}', available inputs: {sorted(obs_dict)}")
            feed[input_info.name] = self._match_input_shape(obs_dict[input_info.name], input_info)
        return feed

    @staticmethod
    def _match_input_shape(value, input_info):
        value = np.asarray(value, dtype=np.float32)
        expected_width = InterMimicRod._expected_input_width(input_info)
        if expected_width is None or value.ndim != 2 or value.shape[1] == expected_width:
            return value

        if value.shape[1] > expected_width:
            return value[:, :expected_width].astype(np.float32)

        padded = np.zeros((value.shape[0], expected_width), dtype=np.float32)
        padded[:, :value.shape[1]] = value
        return padded

    @staticmethod
    def _expected_input_width(input_info):
        shape = input_info.shape
        if len(shape) < 2 or not isinstance(shape[1], int):
            return None
        return shape[1]

    @staticmethod
    def _add_wrist_6dof(action):
        action = np.insert(action, 19, [0.0, 0.0, 0.0])
        return np.insert(action, 26, [0.0, 0.0, 0.0])
