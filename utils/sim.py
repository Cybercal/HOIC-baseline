import numpy as np

from .math_utils import quat_rotate_inverse_xyzw


def get_obs_from_mujoco(data, num_actuators):
    q = data.qpos.astype(np.float32)
    dq = data.qvel.astype(np.float32)
    sensor_dim = 3 * num_actuators

    if data.sensordata.shape[0] >= sensor_dim + 7:
        quat_wxyz = data.sensordata[sensor_dim:sensor_dim + 4].astype(np.float32)
        ang_vel = data.sensordata[sensor_dim + 4:sensor_dim + 7].astype(np.float32)
        qj = data.sensordata[:num_actuators].astype(np.float32)
        dqj = data.sensordata[num_actuators:2 * num_actuators].astype(np.float32)
    else:
        quat_wxyz = q[3:7]
        ang_vel = dq[3:6]
        qj = q[7:7 + num_actuators]
        dqj = dq[6:6 + num_actuators]

    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)
    lin_vel_body = quat_rotate_inverse_xyzw(quat_xyzw, dq[:3])
    return {
        "qj": qj,
        "dqj": dqj,
        "quat": quat_xyzw,
        "ang_vel": ang_vel,
        "lin_vel": lin_vel_body,
        "pos": q[:3],
    }


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd
