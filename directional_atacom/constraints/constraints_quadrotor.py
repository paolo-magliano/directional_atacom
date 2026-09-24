import numpy as np

from air_hockey_challenge.constraints import Constraint

# Those constraints implements intrinsically the vialibity computation so they don't need to compute it in the atacom alg
# even if the quadrotor is a second order system

class ZPositionConstraint(Constraint):
    def __init__(self, env_info, **kwargs):
        super().__init__(env_info, 2, **kwargs)
        self._name = 'z_pos_constr'
        self.zeta_vertical = 0.3

    def _fun(self, q, dq):
        z, dz = q[2], dq[2]
        k = np.array([z, -z])

        dk = np.array([dz, -dz])
        zetas = np.array([self.zeta_vertical, self.zeta_vertical])

        bounds = np.array([-self._env_info['bounds'][5], self._env_info['bounds'][4]])
        self._fun_value = k + zetas * dk + bounds
        return self._fun_value

    def _jacobian(self, q, dq):
        self._jac_value[0, 2] = 1
        self._jac_value[0, 9] = self.zeta_vertical
        self._jac_value[1, 2] = -1
        self._jac_value[1, 9] = -self.zeta_vertical

        return self._jac_value

class ZRotationConstraint(Constraint):
    def __init__(self, env_info, **kwargs):
        super().__init__(env_info, 2, **kwargs)  
        self._name = 'z_rot_constr'
        self.z_limit = 0.1

    def _fun(self, q, dq):
        w_z = dq[6]

        self._fun_value = np.array([w_z - self.z_limit, - w_z + self.z_limit])
        return self._fun_value

    def _jacobian(self, q, dq):
        self._jac_value[0, 13] = 1
        self._jac_value[1, 13] = -1

        return self._jac_value

class WallConstraint(Constraint):
    # Orientation Only Constraint
    # 2 * (qw * qy + qx * qz) + zeta * 2 * [qy, qz, qw, qx] * [dqw, dqx, dqy, dqz] - lambda * (u - x - zeta * dx) < 0
    # - 2 * (qw * qy + qx * qz) + zeta * (-2 * [qy, qz, qw, qx] * [dqw, dqx, dqy, dqz]) - lambda * (-l + x + zeta * dx) < 0
    def __init__(self, env_info, **kwargs):
        super().__init__(env_info, 4, **kwargs)  
        self._name = 'wall_constr'
        self.zeta_horizontal = 2.
        self.zeta_rotational = 0.5

    def _fun(self, q, dq):
        x, dx = q[0], dq[0]
        y, dy = q[1], dq[1]
        qw, qx, qy, qz = q[3:7]
        _, w_x, w_y, w_z = dq[3:7]
        dquat = quat_mult(np.array([0, w_x, w_y, w_z]), np.array([qw, qx, qy, qz])) / 2

        thrust_x = 2 * (qw * qy + qx * qz)
        thrust_y = 2 * (qy * qz - qw * qx)
        dthrust_x = _jac_quat_x(qw, qx, qy, qz) @ dquat
        dthrust_y = _jac_quat_y(qw, qx, qy, qz) @ dquat
        k = np.array([thrust_x, -thrust_x, thrust_y, -thrust_y])
        dk = np.array([dthrust_x, -dthrust_x, dthrust_y, -dthrust_y])
        zetas = np.array([self.zeta_rotational, self.zeta_rotational,
                          self.zeta_rotational, self.zeta_rotational])

        angle_bounds = np.array([x + self.zeta_horizontal * dx - self._env_info['bounds'][1] + self._env_info['constr_margin'], - x - self.zeta_horizontal * dx + self._env_info['bounds'][0] + self._env_info['constr_margin'],
                                 y + self.zeta_horizontal * dy - self._env_info['bounds'][3] + self._env_info['constr_margin'], - y - self.zeta_horizontal * dy + self._env_info['bounds'][2] + self._env_info['constr_margin']])
        angle_bounds = np.maximum(angle_bounds, -0.1)

        self._fun_value = k + zetas * dk + angle_bounds
        return self._fun_value

    def _jacobian(self, q, dq):
        x, dx = q[0], dq[0]
        y, dy = q[1], dq[1]
        qw, qx, qy, qz = q[3:7]
        _, w_x, w_y, w_z = dq[3:7]
        dquat = quat_mult(np.array([0, w_x, w_y, w_z]), np.array([qw, qx, qy, qz])) / 2

        # Thrust X
        self._jac_value[0, 3:7] = _jac_quat_x(qw, qx, qy, qz) + self.zeta_rotational * \
            _jac_quat_x_dot(dquat[0], dquat[1], dquat[2], dquat[3])
        self._jac_value[0, 11:14] = self.zeta_rotational * _jac_quat_x(qw, qx, qy, qz) @ _W_prime(qw, qx, qy, qz).T / 2

        self._jac_value[1, 3:7] = -_jac_quat_x(qw, qx, qy, qz) - self.zeta_rotational * \
            _jac_quat_x_dot(dquat[0], dquat[1], dquat[2], dquat[3])
        self._jac_value[1, 11:14] = -self.zeta_rotational * _jac_quat_x(qw, qx, qy, qz) @ _W_prime(qw, qx, qy, qz).T / 2

        # Thrust Y
        self._jac_value[2, 3:7] = _jac_quat_y(qw, qx, qy, qz) + self.zeta_rotational * \
            _jac_quat_y_dot(dquat[0], dquat[1], dquat[2], dquat[3])
        self._jac_value[2, 11:14] = self.zeta_rotational * _jac_quat_y(qw, qx, qy, qz) @ _W_prime(qw, qx, qy, qz).T / 2

        self._jac_value[3, 3:7] = -_jac_quat_y(qw, qx, qy, qz) - self.zeta_rotational * \
            _jac_quat_y_dot(dquat[0], dquat[1], dquat[2], dquat[3])
        self._jac_value[3, 11:14] = -self.zeta_rotational * _jac_quat_y(qw, qx, qy, qz) @ _W_prime(qw, qx, qy, qz).T / 2

        return self._jac_value

class XYPointsConstraint(Constraint):
    def __init__(self, env_info, **kwargs):
        super().__init__(env_info, env_info['points'].shape[0], **kwargs)  
        self._name = 'points_constr'
        self.zeta_horizontal = 1.
        self.zeta_rotational = 0.5

    def _fun(self, q, dq):
        x, dx = q[0], dq[0]
        y, dy = q[1], dq[1]
        qw, qx, qy, qz = q[3:7]
        _, w_x, w_y, w_z = dq[3:7]
        dquat = quat_mult(np.array([0, w_x, w_y, w_z]), np.array([qw, qx, qy, qz])) / 2

        thrust_x = 2 * (qw * qy + qx * qz)
        thrust_y = 2 * (qy * qz - qw * qx)
        dthrust_x = _jac_quat_x(qw, qx, qy, qz) @ dquat
        dthrust_y = _jac_quat_y(qw, qx, qy, qz) @ dquat

        k_list = []
        for px, py, pr in self._env_info['points']:
            x_dist = x - px
            y_dist = y - py
            dist = np.sqrt(x_dist ** 2 + y_dist ** 2) + 1e-8

            dist_margin = - dist + pr + self._env_info['constr_margin']

            dx_dist = - x_dist / dist
            dy_dist = - y_dist / dist

            ddist = np.array([dx_dist, dy_dist]).T @ np.array([dx, dy])

            dist_thrust = np.array([dx_dist, dy_dist]).T @ np.array([thrust_x, thrust_y])

            ddist_thrust = np.array([dx_dist, dy_dist]).T @ np.array([dthrust_x, dthrust_y])

            bound = dist_margin + self.zeta_horizontal * ddist

            bound = np.maximum(bound, -0.1)

            k = dist_thrust 

            dk = ddist_thrust

            k_viability = k + self.zeta_rotational * dk + bound
            k_list.append(k_viability)
        self._fun_value = np.array(k_list)
        return self._fun_value

    def _jacobian(self, q, dq):
        x, dx = q[0], dq[0]
        y, dy = q[1], dq[1]
        qw, qx, qy, qz = q[3:7]
        _, w_x, w_y, w_z = dq[3:7]
        dquat = quat_mult(np.array([0, w_x, w_y, w_z]), np.array([qw, qx, qy, qz])) / 2

        for i, (px, py, pr) in enumerate(self._env_info['points']):
            j_point = np.zeros_like(self._jac_value[0])

            x_dist = x - px
            y_dist = y - py
            dist = np.sqrt(x_dist ** 2 + y_dist ** 2) + 1e-8 

            dx_dist = - x_dist / dist
            dy_dist = - y_dist / dist

            ddist = np.array([dx_dist, dy_dist]).T @ np.array([dx, dy])

            j_thrust_x = _jac_quat_x(qw, qx, qy, qz) + self.zeta_rotational * \
                _jac_quat_x_dot(dquat[0], dquat[1], dquat[2], dquat[3])

            j_dthrust_x = self.zeta_rotational * _jac_quat_x(qw, qx, qy, qz) @ _W_prime(qw, qx, qy, qz).T / 2

            j_thrust_y = _jac_quat_y(qw, qx, qy, qz) + self.zeta_rotational * \
                _jac_quat_y_dot(dquat[0], dquat[1], dquat[2], dquat[3])

            j_dthrust_y = self.zeta_rotational * _jac_quat_y(qw, qx, qy, qz) @ _W_prime(qw, qx, qy, qz).T / 2

            j_point[3:7] = np.array([dx_dist, dy_dist]).T @ np.array([j_thrust_x, j_thrust_y])

            j_point[11:14] = np.array([dx_dist, dy_dist]).T @ np.array([j_dthrust_x, j_dthrust_y])

            self._jac_value[i, :] = j_point
        return self._jac_value

def quat_mult(q, p):
    q0, q1, q2, q3 = q
    Q_q = np.array([[q0, -q1, -q2, -q3],
                    [q1, q0, q3, -q2],
                    [q2, -q3, q0, q1],
                    [q3, q2, -q1, q0]])

    return Q_q @ p

def _jac_quat_x(qw, qx, qy, qz):
    return np.array([qy, qz, qw, qx]) * 2

def _jac_quat_y(qw, qx, qy, qz):
    return np.array([-qx, -qw, qz, qy]) * 2

def _jac_quat_x_dot(qw_dot, qx_dot, qy_dot, qz_dot):
    return np.array([qy_dot, qz_dot, qw_dot, qx_dot]) * 2

def _jac_quat_y_dot(qw_dot, qx_dot, qy_dot, qz_dot):
    return np.array([-qx_dot, -qw_dot, qz_dot, qy_dot]) * 2

def _W_prime(qw, qx, qy, qz):
    return np.array([[-qx, qw, qz, -qy], [-qy, -qz, qw, qx], [-qz, qy, -qx, qw]])