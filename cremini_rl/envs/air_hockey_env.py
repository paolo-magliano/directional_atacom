import torch
import mujoco
import numpy as np
from enum import Enum
from mushroom_rl.utils.spaces import Box
from air_hockey_challenge.environments.iiwas import AirHockeySingle
from air_hockey_challenge.constraints import JointPositionConstraint, JointVelocityConstraint, EndEffectorConstraint, LinkConstraint
from collections import OrderedDict

from cremini_rl.constraints.constraints import ConstraintCollection as ConstraintList
from cremini_rl.utils.control import VelocityControl, AccelerationControl
from cremini_rl.envs.planar_air_hockey_env import AbsorbType, Cache

class AirHockey(AirHockeySingle):
    def __init__(self, return_cost=True, headless=True):
        self.return_cost = return_cost

        # if headless:
        super().__init__(horizon=200, viewer_params={'headless': headless, 'camera_params': {
                            'static': dict(distance=5.0, elevation=-45.0, azimuth=90.0, lookat=np.array([0.0, 0.0, 0.0])),
                            'follow': dict(distance=3.5, elevation=0.0, azimuth=90.0),
                            'top_static': dict(distance=5.0, elevation=-90.0, azimuth=90.0, lookat=np.array([0.0, 0.0, 0.0]))
                        }})

        # Compute Link Constraint Bound
        x_l = - self.env_info['robot']['base_frame'][0][0, 3] - (
                self.env_info['table']['length'] / 2 - self.env_info['mallet']['radius'])
        y_l = - (self.env_info['table']['width'] / 2 - self.env_info['mallet']['radius'])
        y_u = self.env_info['table']['width'] / 2 - self.env_info['mallet']['radius']
        z_l = self.env_info['robot']['ee_desired_height'] - 0.02
        z_u = self.env_info['robot']['ee_desired_height'] + 0.02
        z_wr = 0.25
        z_el = 0.25
        self.link_constr_ub = np.array([x_l, y_l, -y_u, z_wr, z_el])
        self.ee_height_ub = np.array([z_l, -z_u])

        self.ee_puck_dist = np.inf
        self.constr_dim = 1

        self._end_effector_const = EndEffectorConstraint(self.env_info)

        self.info.action_space = Box(low=-np.ones(self.env_info['robot']['n_joints']), high=np.ones(self.env_info['robot']['n_joints']))

    def constraint_init(self, constraints_class, K_values):
        self.K = []
        self.original_constraint_list = ConstraintList()

        for constr_class, k in zip(constraints_class, K_values):
            constr = constr_class(self.env_info)
            self.original_constraint_list.add(constr)
            self.K += [k] * constr.output_dim

        self.K = np.array(self.K)

    def step(self, action):
        obs, reward, done, info = super(AirHockey, self).step(action)

        cost = info["cost"]

        if self.return_cost:
            return obs, reward, cost, done, info
        return obs, reward, done, info

    def _compute_ee_height_constraint(self, q):
        q_pos = q[:7]
        q_vel = q[7:]
        # Only take z height endeffector constraints
        cons = self._end_effector_const.fun(q_pos, q_vel)[3:]
        J_q = self._end_effector_const.jacobian(q_pos, q_vel)[3:]

        return cons, J_q

    def setup(self, obs=None):
        puck_pos = np.random.uniform([-0.7, -0.4, -np.pi], [-0.3, 0.4, np.pi])
        self._write_data("puck_x_pos", puck_pos[0])
        self._write_data("puck_y_pos", puck_pos[1])
        self._write_data("puck_yaw_pos", puck_pos[2])

        super(AirHockey, self).setup(obs)

        self.absorb_type = AbsorbType.NONE
        self.ee_puck_dist = np.inf

    def reward(self, state, action, next_state, absorbing):
        puck_pos = next_state[:2].copy()
        puck_pos[0] += 1.51
        puck_vel = next_state[3:5]

        if absorbing:
            r = 0
            factor = (1 - self.info.gamma ** self.info.horizon) / (1 - self.info.gamma)
            if self.absorb_type == AbsorbType.GOAL:
                r = 1.5 - (np.clip(abs(puck_pos[1]), 0, 0.1) * 5)
            elif self.absorb_type == AbsorbType.UP:
                r = (1 - np.clip(abs(puck_pos[1]) - 0.1, 0, 0.35) * 2)
            elif self.absorb_type == AbsorbType.LEFT:
                r = (0.3 - np.clip(2.43 - puck_pos[0], 0, 1) * 0.3)
            elif self.absorb_type == AbsorbType.RIGHT:
                r = (0.3 - np.clip(2.43 - puck_pos[0], 0, 1) * 0.3)
            r *= factor
        else:
            r = 0

            ee_pos = self.get_ee()[0][:2] + np.array([1.51, 0.])
            ee_puck_dist = np.linalg.norm(ee_pos - puck_pos)
            if self.ee_puck_dist == np.inf:
                self.ee_puck_dist = ee_puck_dist
            elif ee_puck_dist < self.ee_puck_dist:
                r += (self.ee_puck_dist - ee_puck_dist) * 10
                self.ee_puck_dist = ee_puck_dist

            if puck_pos[0] > 1.51:
                r += 0.5 * np.clip(puck_vel[0], 0, 3)

        return r

    def is_absorbing(self, obs):
        puck_pos = obs[:2].copy()
        puck_pos[0] += 1.51
        puck_vel = obs[3:5].copy()

        if puck_pos[0] < 0.58 or (puck_pos[0] < 0.63 and puck_vel[0] > 0.):
            self.absorb_type = AbsorbType.BOTTOM
            return True

        if puck_pos[0] > 2.43 or (puck_pos[0] > 2.39 and puck_vel[0] < 0):
            self.absorb_type = AbsorbType.UP
            if abs(puck_pos[1]) < 0.1:
                self.absorb_type = AbsorbType.GOAL
            return True

        if puck_vel[0] > 0. and puck_pos[0] > 1.51:
            if (puck_pos[1] > 0.45 and puck_vel[1] > 0.) or (puck_pos[1] > 0.42 and puck_vel[1] < 0.):
                self.absorb_type = AbsorbType.LEFT
                return True
            if (puck_pos[1] < -0.45 and puck_vel[0] < 0.) or (puck_pos[1] < -0.42 and puck_vel[1] > 0.):
                self.absorb_type = AbsorbType.RIGHT
                return True
        return False

    def _create_info_dictionary(self, state):
        q, dq = self.get_joints(state)
        ee_pos, ee_vel = self.get_ee()
        puck_pos, puck_vel = self.get_puck(state)
        q_max = np.concatenate([-q + self.env_info['robot']['joint_pos_limit'][0] * 0.95,
                                q - self.env_info['robot']['joint_pos_limit'][1] * 0.95]).max()

        pos_offset = self.env_info['robot']['base_frame'][0][:3, 3]
        ee_pos = self._data.body("iiwa_1/striker_mallet").xpos - pos_offset
        wr_pos = self._data.body("iiwa_1/link_6").xpos - pos_offset
        el_pos = self._data.body("iiwa_1/link_4").xpos - pos_offset

        link_max = np.array([-ee_pos[0], -ee_pos[1], ee_pos[1], -wr_pos[2], -el_pos[2]]) + self.link_constr_ub

        # ee_height_max = np.array([-ee_pos[2], ee_pos[2]]) + self.ee_height_ub

        cost = max([q_max, link_max.max()])
        success = False
        if self.absorb_type == AbsorbType.GOAL:
            success = True

        return {'cost': cost, 'success': success,  'q_cost': q_max, 'link_cost': link_max.max(),
                'puck_vel': np.linalg.norm(puck_vel), 'joint_vel': np.linalg.norm(dq)}

    def constraint_func(self, q):
        N = len(q)
        cons = np.zeros((N, self.original_constraint_list.output_dim()))
        J_q = np.zeros((N, self.original_constraint_list.output_dim(), q.shape[-1]))
        for i in range(N):
            c, J = self._original_constraint(q[i])

            cons[i] = c
            J_q[i] = J[..., :q.shape[-1]]

        return cons, J_q, np.zeros((N, self.original_constraint_list.output_dim(), 0)), self.K

    def _original_constraint(self, q):
        pos = q[:7]
        vel = q[7:] if len(q) > 7 else np.zeros(7)

        constraint_keys = self.original_constraint_list.keys()
        constraints = []
        constraints_J = []

        for key in constraint_keys:
            constraints.append(self.original_constraint_list.get(key).fun(pos, vel))
            constraints_J.append(self.original_constraint_list.get(key).jacobian(pos, vel).copy())

        const = np.concatenate(constraints)
        J_q = np.vstack(constraints_J)

        return const, J_q

class AirHockeyVel(VelocityControl, AirHockey):
    def __init__(self, return_cost=True, headless=True):
        p_gain = [1500., 1500., 1200., 1200., 1000., 1000., 500.]
        d_gain = [60, 80, 60, 30, 10, 1, 0.5]
        i_gain = [0, 0, 0, 0, 0, 0, 0]

        AirHockey.__init__(self, return_cost, headless)
        VelocityControl.__init__(self, p_gain=p_gain, d_gain=d_gain, i_gain=i_gain)

        constraints_class = [JointPositionConstraint, EndEffectorConstraint, LinkConstraint]
        K_values = [1.0, 0.5, 0.5]

        self.constraint_init(constraints_class, K_values)

class AirHockeyAcc(AccelerationControl, AirHockey):
    def __init__(self, return_cost=True, headless=True):
        super(AirHockeyAcc, self).__init__(return_cost, headless)

        constraints_class = [JointPositionConstraint, JointVelocityConstraint, EndEffectorConstraint, LinkConstraint]
        K_values = [1.0, 0.0, 0.5, 0.5]

        self.constraint_init(constraints_class, K_values)

    def _preprocess_action(self, action):
        action = super(AirHockeyAcc, self)._preprocess_action(action)
        return action * self.env_info['robot']['joint_acc_limit'][1]

    def _create_info_dictionary(self, state):
        q, dq = self.get_joints(state)

        dq_max = np.concatenate([-dq + self.env_info['robot']['joint_vel_limit'][0] * 0.95,
                                 dq - self.env_info['robot']['joint_vel_limit'][1] * 0.95]).max()

        info = super(AirHockeyAcc, self)._create_info_dictionary(state)

        cost = max([dq_max, info['cost']])

        info.update({'cost': cost, 'dq_cost': dq_max})

        return info
