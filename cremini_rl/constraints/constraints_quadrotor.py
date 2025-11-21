from cremini_rl.constraints.constraints import ConstraintBase as Constraint
import numpy as np

def quat_mult(q, p):
    q0, q1, q2, q3 = q
    Q_q = np.array([[q0, -q1, -q2, -q3],
                    [q1, q0, q3, -q2],
                    [q2, -q3, q0, q1],
                    [q3, q2, -q1, q0]])

    return Q_q @ p

class WallConstraint(Constraint):
    # Orientation Only Constraint
    # 2 * (qw * qy + qx * qz) + zeta * 2 * [qy, qz, qw, qx] * [dqw, dqx, dqy, dqz] - lambda * (u - x - zeta * dx) < 0
    # - 2 * (qw * qy + qx * qz) + zeta * (-2 * [qy, qz, qw, qx] * [dqw, dqx, dqy, dqz]) - lambda * (-l + x + zeta * dx) < 0
    def __init__(self, env_info, bound=[-0.98, 0.98, -0.98, 0.98], **kwargs):
        self.bound = bound
        self.zeta_horizontal = 1.6
        self.zeta_rotational = 0.5

        super().__init__(env_info, 4, 13, **kwargs)

    def fun(self, q, z=None) -> np.ndarray:
        x, dx = q[0], q[7]
        y, dy = q[1], q[8]
        qw, qx, qy, qz = q[3:7]
        omega_x, omega_y, omega_z = q[10:13]
        dquat = quat_mult(np.array([0, omega_x, omega_y, omega_z]), np.array([qw, qx, qy, qz])) / 2

        thrust_x = 2 * (qw * qy + qx * qz)
        thrust_y = 2 * (qy * qz - qw * qx)
        dthrust_x = self._jac_quat_x(qw, qx, qy, qz) @ dquat
        dthrust_y = self._jac_quat_y(qw, qx, qy, qz) @ dquat
        k = np.array([thrust_x, -thrust_x, thrust_y, -thrust_y])
        dk = np.array([dthrust_x, -dthrust_x, dthrust_y, -dthrust_y])
        zetas = np.array([self.zeta_rotational, self.zeta_rotational,
                          self.zeta_rotational, self.zeta_rotational])

        angle_bounds = -np.array([self.bound[1] - x - self.zeta_horizontal * dx, - self.bound[0] + x + self.zeta_horizontal * dx,
                                 self.bound[3] - y - self.zeta_horizontal * dy, - self.bound[0] + y + self.zeta_horizontal * dy])
        angle_bounds = np.clip(angle_bounds, -0.1, 0.1)

        k_viability = k + zetas * dk + angle_bounds
        return k_viability

    def jacobian(self, q, z=None) -> np.ndarray:
        qw, qx, qy, qz = q[3:7]
        omega_x, omega_y, omega_z = q[10:13]
        dquat = quat_mult(np.array([0, omega_x, omega_y, omega_z]), np.array([qw, qx, qy, qz])) / 2
        J = np.zeros((self.output_dim, self.n_joints))

        # Thrust X
        J[0, 3:7] = self._jac_quat_x(qw, qx, qy, qz) + self.zeta_rotational * \
            self._jac_quat_x_dot(dquat[0], dquat[1], dquat[2], dquat[3])
        J[0, 10:13] = self.zeta_rotational * self._jac_quat_x(qw, qx, qy, qz) @ self._W_prime(qw, qx, qy, qz).T / 2
        J[1, 3:7] = -self._jac_quat_x(qw, qx, qy, qz) - self.zeta_rotational * \
            self._jac_quat_x_dot(dquat[0], dquat[1], dquat[2], dquat[3])
        J[1, 10:13] = -self.zeta_rotational * self._jac_quat_x(qw, qx, qy, qz) @ self._W_prime(qw, qx, qy, qz).T / 2

        # Thrust Y
        J[2, 3:7] = self._jac_quat_y(qw, qx, qy, qz) + self.zeta_rotational * \
            self._jac_quat_y_dot(dquat[0], dquat[1], dquat[2], dquat[3])
        J[2, 10:13] = self.zeta_rotational * self._jac_quat_y(qw, qx, qy, qz) @ self._W_prime(qw, qx, qy, qz).T / 2
        J[3, 3:7] = -self._jac_quat_y(qw, qx, qy, qz) - self.zeta_rotational * \
            self._jac_quat_y_dot(dquat[0], dquat[1], dquat[2], dquat[3])
        J[3, 10:13] = -self.zeta_rotational * self._jac_quat_y(qw, qx, qy, qz) @ self._W_prime(qw, qx, qy, qz).T / 2

        return J

    def _jac_quat_x(self, qw, qx, qy, qz):
        return np.array([qy, qz, qw, qx]) * 2

    def _jac_quat_y(self, qw, qx, qy, qz):
        return np.array([-qx, -qw, qz, qy]) * 2

    def _jac_quat_x_dot(self, qw_dot, qx_dot, qy_dot, qz_dot):
        return np.array([qy_dot, qz_dot, qw_dot, qx_dot]) * 2

    def _jac_quat_y_dot(self, qw_dot, qx_dot, qy_dot, qz_dot):
        return np.array([-qx_dot, -qw_dot, qz_dot, qy_dot]) * 2

    def _W_prime(self, qw, qx, qy, qz):
        return np.array([[-qx, qw, qz, -qy], [-qy, -qz, qw, qx], [-qz, qy, -qx, qw]])

class ZPositionConstraint(Constraint):
    def __init__(self, env_info, z_bound=[-0.98, 0.1], **kwargs):
        self.z_bound = z_bound

        self.zeta_vertical = 0.3
        super().__init__(env_info, 2, 13, **kwargs)

    def fun(self, q, z=None) -> np.ndarray:
        z, dz = q[2], q[9]
        k = np.array([z, -z])

        dk = np.array([dz, -dz])
        zetas = np.array([self.zeta_vertical, self.zeta_vertical])

        bounds = np.array(self.z_bound)
        k_viability = k + zetas * dk + bounds
        return k_viability

    def jacobian(self, q, z=None) -> np.ndarray:
        J = np.zeros((self.output_dim, self.n_joints))

        # Z direction
        J[0, 2] = 1
        J[0, 9] = self.zeta_vertical
        J[1, 2] = -1
        J[1, 9] = -self.zeta_vertical

        return J

class XYPointsConstraint(Constraint):
    def __init__(self, env_info, points=[[0.5, 0.5], [-0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]], r_margin=0.1, **kwargs):
        self.points = points
        self.r_margin = r_margin

        self.zeta_horizontal = 1.
        self.zeta_rotational = 0.5

        super().__init__(env_info, 4 * len(points), 13, **kwargs)

    def fun(self, q, z=None) -> np.ndarray:
        x, dx = q[0], q[7]
        y, dy = q[1], q[8]
        qw, qx, qy, qz = q[3:7]
        omega_x, omega_y, omega_z = q[10:13]
        dquat = quat_mult(np.array([0, omega_x, omega_y, omega_z]), np.array([qw, qx, qy, qz])) / 2

        thrust_x = 2 * (qw * qy + qx * qz)
        thrust_y = 2 * (qy * qz - qw * qx)
        dthrust_x = self._jac_quat_x(qw, qx, qy, qz) @ dquat
        dthrust_y = self._jac_quat_y(qw, qx, qy, qz) @ dquat

        k_list = []
        for px, py in self.points:
            dist = np.sqrt((x - px) ** 2 + (y - py) ** 2) - self.r_margin + \
                np.sign((x - px)) * self.zeta_horizontal * dx + \
                np.sign((y - py)) * self.zeta_horizontal * dy

            bound = np.clip(dist, 0, 0.1)

            k = np.array([thrust_x, -thrust_x, thrust_y, -thrust_y])

            dk = np.array([dthrust_x, -dthrust_x, dthrust_y, -dthrust_y])

            zetas = np.array([self.zeta_rotational, self.zeta_rotational,
                                self.zeta_rotational, self.zeta_rotational])

            bounds = np.array([-bound, -bound, -bound, -bound]) 

            k_viability = k + zetas * dk + bounds
            k_list.append(k_viability)
        return np.concatenate(k_list, axis=0)

    def jacobian(self, q, z=None) -> np.ndarray:
        qw, qx, qy, qz = q[3:7]
        omega_x, omega_y, omega_z = q[10:13]
        dquat = quat_mult(np.array([0, omega_x, omega_y, omega_z]), np.array([qw, qx, qy, qz])) / 2

        J_single = np.zeros((4, self.n_joints))

        # Thrust X
        J_single[0, 3:7] = self._jac_quat_x(qw, qx, qy, qz) + self.zeta_rotational * \
            self._jac_quat_x_dot(dquat[0], dquat[1], dquat[2], dquat[3])
        J_single[0, 10:13] = self.zeta_rotational * self._jac_quat_x(qw, qx, qy, qz) @ self._W_prime(qw, qx, qy, qz).T / 2
        J_single[1, 3:7] = -self._jac_quat_x(qw, qx, qy, qz) - self.zeta_rotational * \
            self._jac_quat_x_dot(dquat[0], dquat[1], dquat[2], dquat[3])
        J_single[1, 10:13] = -self.zeta_rotational * self._jac_quat_x(qw, qx, qy, qz) @ self._W_prime(qw, qx, qy, qz).T / 2

        # Thrust Y
        J_single[2, 3:7] = self._jac_quat_y(qw, qx, qy, qz) + self.zeta_rotational * \
            self._jac_quat_y_dot(dquat[0], dquat[1], dquat[2], dquat[3])
        J_single[2, 10:13] = self.zeta_rotational * self._jac_quat_y(qw, qx, qy, qz) @ self._W_prime(qw, qx, qy, qz).T / 2
        J_single[3, 3:7] = -self._jac_quat_y(qw, qx, qy, qz) - self.zeta_rotational * \
            self._jac_quat_y_dot(dquat[0], dquat[1], dquat[2], dquat[3])
        J_single[3, 10:13] = -self.zeta_rotational * self._jac_quat_y(qw, qx, qy, qz) @ self._W_prime(qw, qx, qy, qz).T / 2
        J = np.zeros((self.output_dim, self.n_joints))
        for i in range(len(self.points)):
            J[i * 4:(i + 1) * 4, :] = J_single
        return J

    def _jac_quat_x(self, qw, qx, qy, qz):
        return np.array([qy, qz, qw, qx]) * 2

    def _jac_quat_y(self, qw, qx, qy, qz):
        return np.array([-qx, -qw, qz, qy]) * 2

    def _jac_quat_x_dot(self, qw_dot, qx_dot, qy_dot, qz_dot):
        return np.array([qy_dot, qz_dot, qw_dot, qx_dot]) * 2

    def _jac_quat_y_dot(self, qw_dot, qx_dot, qy_dot, qz_dot):
        return np.array([-qx_dot, -qw_dot, qz_dot, qy_dot]) * 2

    def _W_prime(self, qw, qx, qy, qz):
        return np.array([[-qx, qw, qz, -qy], [-qy, -qz, qw, qx], [-qz, qy, -qx, qw]])
      