import torch
import mujoco
import numpy as np
from enum import Enum
from mushroom_rl.utils.spaces import Box
from air_hockey_challenge.environments.iiwas import AirHockeySingle
from air_hockey_challenge.constraints import JointPositionConstraint, JointVelocityConstraint
from air_hockey_challenge.utils.kinematics import forward_kinematics, jacobian
from collections import OrderedDict

from cremini_rl.envs.base_constr_env import ConstrEnv
from cremini_rl.constraints.constraints import ConstraintCollection as ConstraintList
from cremini_rl.utils.control import VelocityControl, AccelerationControl
from cremini_rl.envs.planar_air_hockey_env import AbsorbType
from cremini_rl.constraints.constraints_air_hockey import EndEffectorConstraint, LinkConstraint

class AirHockey(AirHockeySingle, ConstrEnv):
    def __init__(self, return_cost=True, headless=True, action_filter_ratio=None):
        super().__init__(horizon=200, viewer_params={'headless': headless, 'width': 1920, 'height': 1080, 'camera_params': {
                            'static': dict(distance=5.0, elevation=-45.0, azimuth=90.0, lookat=np.array([0.0, 0.0, 0.0])),
                            'follow': dict(distance=3.5, elevation=0.0, azimuth=90.0),
                            'top_static': dict(distance=5.0, elevation=-90.0, azimuth=90.0, lookat=np.array([0.0, 0.0, 0.0]))
                        }})

        self.return_cost = return_cost

        self._model.geom("iiwa_1/ee").solref = np.array([0.02, 0.8])

        # Compute Link Constraint Bound
        x_l = - self.env_info['robot']['base_frame'][0][0, 3] - (
                self.env_info['table']['length'] / 2 - self.env_info['mallet']['radius'])
        y_l = - (self.env_info['table']['width'] / 2 - self.env_info['mallet']['radius'])
        y_u = self.env_info['table']['width'] / 2 - self.env_info['mallet']['radius']
        z_l = self.env_info['robot']['ee_desired_height'] - 0.02
        z_u = self.env_info['robot']['ee_desired_height'] + 0.02
        z_wr = 0.35
        z_el = 0.35
        
        self.link_constr_ub = np.array([x_l, y_l, -y_u, z_wr, z_el])
        self.ee_height_ub = np.array([z_l, -z_u])

        self.constr_dim = 1

        self.prev_action = np.zeros(self.info.action_space.shape)
        self.action_filter_ratio = action_filter_ratio

        self.info.action_space = Box(low=-np.ones(self.env_info['robot']['n_joints']), high=np.ones(self.env_info['robot']['n_joints']))  

        self.integrator_pos_limit = self.env_info['robot']['joint_pos_limit'] * 0.95

        self.env_info['robot']['joint_pos_limit'][:, 2] = np.array([-np.pi/2, np.pi/2])
        self.env_info['robot']['joint_pos_limit'][:, 4] = np.array([-np.pi/2, np.pi/2])
        self.env_info['robot']['joint_pos_limit'][:, 6] = np.array([-np.pi/2, np.pi/2])

        self.modify_mdp_info()

        self.puck_vel_cross = 0.

    def modify_mdp_info(self):
        # Include previous action
        if self.action_filter_ratio:
            obs_low = np.concatenate([self.info.observation_space.low, -np.ones(self.info.action_space.shape)])
            obs_high = np.concatenate([self.info.observation_space.high, np.ones(self.info.action_space.shape)])
            self.info.observation_space = Box(obs_low, obs_high)

        # Include ee info
        obs_low = np.concatenate([self.info.observation_space.low, [-2., -1., -1., -1., -5., -5.]])
        obs_high = np.concatenate([self.info.observation_space.high, [2., 1., 1., 1., 5., 5.]])

        self.info.observation_space = Box(obs_low, obs_high)

    def _modify_observation(self, obs):
        obs = super(AirHockey, self)._modify_observation(obs)
        if self.action_filter_ratio:
            obs = self.add_prev_action_obs(obs)
        obs = self.add_ee_info_obs(obs)

        return obs

    def step(self, action):
        obs, reward, done, info = super(AirHockey, self).step(action)

        cost = info["cost"]

        if self.return_cost:
            return obs, reward, cost, done, info
        return obs, reward, done, info

    def setup(self, obs):
        puck_pos = np.random.uniform([-0.7, -0.4, -np.pi], [-0.2, 0.4, np.pi])

        init_vel_angle = np.random.uniform(-np.pi, np.pi)
        init_vel_mag = np.random.uniform(0, 0.3)
        init_vel = np.array([np.cos(init_vel_angle), np.sin(init_vel_angle)]) * init_vel_mag
        hit_range = np.array([-0.7, -0.2])
        t_reach = np.array([hit_range[0] - puck_pos[0], hit_range[1] - puck_pos[0]]) / init_vel[0]

        if np.any(t_reach < 0):
            # The puck will leave the hit range, Scale down the velocity if it's leaving too fast
            init_vel[0] *= np.minimum(np.max(t_reach) / 2., 1.)
        else:
            # The puck is not in the hit range, scale up the velocity if it's reaching too slow
            init_vel[0] *= np.maximum(np.min(t_reach), 1.)

        yaw_vel = np.random.uniform(0, 5)
        init_puck_pos = np.array(puck_pos)
        init_puck_vel = np.array([*init_vel, yaw_vel])

        self._write_data("puck_x_pos", init_puck_pos[0])
        self._write_data("puck_y_pos", init_puck_pos[1])
        self._write_data("puck_yaw_pos", init_puck_pos[2])
        self._write_data("puck_x_vel", init_puck_vel[0])
        self._write_data("puck_y_vel", init_puck_vel[1])
        self._write_data("puck_yaw_vel", init_puck_vel[2])
        super().setup(obs)

        self.absorb_type = AbsorbType.NONE
        self.ee_puck_dist = np.inf
        self.puck_vel_cross = 0.

    def reward(self, state, action, next_state, absorbing):
        puck_pos = next_state[:2].copy()
        puck_pos[0] += 1.51
        puck_vel = next_state[3:5]

        if absorbing:
            r = 0
            factor = (1 - self.info.gamma ** self.info.horizon) / (1 - self.info.gamma)
            if self.absorb_type == AbsorbType.GOAL:
                r = 2 - (np.clip(abs(puck_pos[1]), 0, 0.1) * 5)
            elif self.absorb_type == AbsorbType.UP:
                r = (1 - np.clip(abs(puck_pos[1]) - 0.1, 0, 0.35) * 2)
            elif self.absorb_type == AbsorbType.LEFT:
                r = (0.3 - np.clip(2.43 - puck_pos[0], 0, 1) * 0.3)
            elif self.absorb_type == AbsorbType.RIGHT:
                r = (0.3 - np.clip(2.43 - puck_pos[0], 0, 1) * 0.3)
            r *= factor
        else:
            if puck_pos[0] > 1.51:
                r = 1.5 * np.clip(puck_vel[0], 0, 3)
            else:
                r = 0

        ee_pos = self.get_ee()[0][:2] + np.array([1.51, 0.])
        ee_puck_dist = np.linalg.norm(ee_pos - puck_pos)
        if self.ee_puck_dist == np.inf:
            self.ee_puck_dist = ee_puck_dist
        elif ee_puck_dist < self.ee_puck_dist:
            r += (self.ee_puck_dist - ee_puck_dist) * 10
            self.ee_puck_dist = ee_puck_dist

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

    def save_prev_action_obs(self, action):
        self.prev_action = action.copy()

    def add_prev_action_obs(self, obs):
        return np.concatenate([obs, self.prev_action.squeeze()])

    def add_ee_info_obs(self, obs):
        joint_pos, joint_vel = self.get_joints(obs)
        ee_pos, _ = forward_kinematics(self.env_info["robot"]["robot_model"], self.env_info["robot"]["robot_data"], joint_pos)
        ee_vel = jacobian(self.env_info["robot"]["robot_model"], self.env_info["robot"]["robot_data"], joint_pos) @ joint_vel
        puck_pos, puck_vel = self.get_puck(obs)

        rel_pos = (puck_pos - ee_pos)[:2]
        rel_vel = puck_vel[:2] - ee_vel[:2]
        rel_ang_sin = rel_pos[1] / np.linalg.norm(rel_pos)
        rel_ang_cos = rel_pos[0] / np.linalg.norm(rel_pos)
        return np.concatenate([obs, rel_pos, [rel_ang_sin, rel_ang_cos], rel_vel])

    def _create_info_dictionary(self, state):
        q, dq = self.get_joints(state)
        ee_pos, ee_vel = self.get_ee()
        puck_pos, puck_vel = self.get_puck(state)
        q_constr = np.concatenate([-q + self.env_info['robot']['joint_pos_limit'][0] * 0.95,
                                q - self.env_info['robot']['joint_pos_limit'][1] * 0.95])

        pos_offset = self.env_info['robot']['base_frame'][0][:3, 3]
        ee_pos = self._data.body("iiwa_1/striker_mallet").xpos - pos_offset
        wr_pos = self._data.body("iiwa_1/link_6").xpos - pos_offset
        el_pos = self._data.body("iiwa_1/link_4").xpos - pos_offset

        link_constr = np.array([-ee_pos[0], -ee_pos[1], ee_pos[1], -wr_pos[2], -el_pos[2]]) + self.link_constr_ub

        # ee_height_max = np.array([-ee_pos[2], ee_pos[2]]) + self.ee_height_ub

        if self.puck_vel_cross == 0. and puck_pos[0] > 0:
            self.puck_vel_cross = puck_vel[0]

        success = False
        if self.absorb_type == AbsorbType.GOAL:
            success = True

        return {'cost': np.concatenate([q_constr, link_constr]), 'success': success,  'q_cost': q_constr.max(), 'link_cost': link_constr.max(),
                'puck_vel_cross': self.puck_vel_cross, 'joint_vel': np.linalg.norm(dq)}

class AirHockeyVel(VelocityControl, AirHockey):
    def __init__(self, return_cost=True, headless=True, action_filter_ratio=None):
        p_gain = [1500., 1500., 1200., 1200., 1000., 1000., 500.]
        d_gain = [60, 80, 60, 30, 10, 1, 0.5]
        i_gain = [0, 0, 0, 0, 0, 0, 0]

        super(AirHockeyVel, self).__init__(return_cost=return_cost, headless=headless, action_filter_ratio=action_filter_ratio,
                                            p_gain=p_gain, d_gain=d_gain, i_gain=i_gain)

        constraints_class = [JointPositionConstraint, EndEffectorConstraint, LinkConstraint]
        K_values = [1.0, 0.5, 0.5]

        self.constraint_init(constraints_class, K_values)

class AirHockeyAcc(AccelerationControl, AirHockey):
    def __init__(self, return_cost=True, headless=True, action_filter_ratio=None):
        super(AirHockeyAcc, self).__init__(return_cost, headless, action_filter_ratio)

        constraints_class = [JointPositionConstraint, JointVelocityConstraint, EndEffectorConstraint, LinkConstraint]
        K_values = [1.0, 0.0, 0.5, 0.5]

        self.constraint_init(constraints_class, K_values)

    def _preprocess_action(self, action):
        action = super(AirHockeyAcc, self)._preprocess_action(action)
        return action * self.env_info['robot']['joint_acc_limit'][1]

    def _create_info_dictionary(self, state):
        q, dq = self.get_joints(state)

        dq_constr = np.concatenate([-dq + self.env_info['robot']['joint_vel_limit'][0] * 0.95,
                                 dq - self.env_info['robot']['joint_vel_limit'][1] * 0.95])

        info = super(AirHockeyAcc, self)._create_info_dictionary(state)

        info['cost'] = np.concatenate([info['cost'], dq_constr])

        info.update({'dq_cost': dq_constr.max()})

        return info
