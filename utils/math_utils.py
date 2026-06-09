import numpy as np


def normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def yaw_quat(quat):
    shape = quat.shape
    quat_yaw = quat.copy().reshape(-1, 4)
    qw = quat_yaw[:, 0]
    qx = quat_yaw[:, 1]
    qy = quat_yaw[:, 2]
    qz = quat_yaw[:, 3]
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    quat_yaw[:] = 0.0
    quat_yaw[:, 0] = np.cos(yaw / 2)
    quat_yaw[:, 3] = np.sin(yaw / 2)
    return normalize(quat_yaw).reshape(shape)


def quat_rotate_inverse_wxyz(q, v):
    shape = v.shape
    q = np.asarray(q, dtype=np.float32).reshape(-1, 4)
    v = np.asarray(v, dtype=np.float32).reshape(-1, 3)
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w**2 - 1.0)[:, None]
    b = np.cross(q_vec, v) * q_w[:, None] * 2.0
    c = q_vec * np.sum(q_vec * v, axis=1, keepdims=True) * 2.0
    return (a - b + c).reshape(shape)


def quat_rotate_inverse_xyzw(q, v):
    q = np.asarray(q, dtype=np.float32)
    q_wxyz = np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)
    return quat_rotate_inverse_wxyz(q_wxyz, np.asarray(v, dtype=np.float32))
