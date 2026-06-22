from .dynamics import ControlAffineSystem
import numpy as np


def quat_mult(q, p):
    q0, q1, q2, q3 = q
    Q_q = np.array([[q0, -q1, -q2, -q3],
                    [q1, q0, q3, -q2],
                    [q2, -q3, q0, q1],
                    [q3, q2, -q1, q0]])

    return Q_q @ p

def batch_quat_mult(q, p):
    q0, q1, q2, q3 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    Q_q = np.array([[q0, -q1, -q2, -q3],
                    [q1, q0, q3, -q2],
                    [q2, -q3, q0, q1],
                    [q3, q2, -q1, q0]]).transpose((2, 0, 1))  # (B, 4, 4)

    return np.einsum('bij,bj->bi', Q_q, p)

class QuadrotorControlSystem(ControlAffineSystem):
    # Quaternion-based dynamics is taken from:
    # https://www.diva-portal.org/smash/get/diva2:1010947/FULLTEXT01.pdf
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9802523

    def __init__(self, mass, inertia, gear_ratios, gravity=9.81):
        self.mass = mass
        self.inertia = np.array(inertia)
        self.gear_ratios = np.array(gear_ratios)
        self.gravity = np.array([0, 0, -gravity])
        super().__init__(14, 4, [i for i in range(13)])

    def get_q(self, state):
        q = super().get_q(state)
        return np.concatenate([q[:, :-3], np.zeros((q.shape[0], 1)), q[:, -3:]], axis=1)

    def f(self, q):
        x, y, z = q[:, 0], q[:, 1], q[:, 2]
        quat = q[:, 3:7]
        vx, vy, vz = q[:, 7], q[:, 8], q[:, 9]
        w_x, w_y, w_z = q[:, 11], q[:, 12], q[:, 13]

        f = np.zeros((q.shape[0], 14))
        f[:, 0: 3] = np.array([vx, vy, vz]).T
        f[:, 3: 7] = np.array([quat_mult(np.array([0, w_x[i], w_y[i], w_z[i]]), quat[i]) for i in range(q.shape[0])]) / 2
        f[:, 7: 10] = self.gravity[None]
        w = np.array([w_x, w_y, w_z]).T
        f[:, 11: 14] = -np.cross(w, self.inertia * w) / self.inertia
        return f[..., None]

    def G(self, q):
        G_mat = np.zeros((q.shape[0], 14, 4))
        qw, qx, qy, qz = q[:, 3], q[:, 4], q[:, 5], q[:, 6]
        G_mat[:, 7: 10, 0] = np.array([2 * (qw * qy + qx * qz),
                                    2 * (qy * qz - qw * qx),
                                    qw ** 2 - qx ** 2 - qy ** 2 + qz ** 2]).T / self.mass
        G_mat[:, 11: 14, 1:] = np.diag(self.gear_ratios / self.inertia)[None]
        return G_mat
