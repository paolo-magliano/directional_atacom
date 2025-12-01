import torch
import mujoco
import numpy as np
from enum import Enum
from mushroom_rl.utils.spaces import Box
from air_hockey_challenge.environments.planar.single import AirHockeySingle as PlanarAirHockeySingle
from air_hockey_challenge.constraints import JointPositionConstraint, JointVelocityConstraint, EndEffectorConstraint
from collections import OrderedDict

from cremini_rl.constraints.constraints import ConstraintCollection as ConstraintList
from cremini_rl.utils.control import VelocityControl, AccelerationControl

class AbsorbType(Enum):
    NONE = 0
    GOAL = 1
    UP = 2
    RIGHT = 3
    LEFT = 4
    BOTTOM = 5


class Cache(OrderedDict):
    def __init__(self, maxsize=200000, /, *args, **kwds):
        self.maxsize = maxsize
        super().__init__(*args, **kwds)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        return value

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            oldest = next(iter(self))
            del self[oldest]

class PlanarAirHockey(PlanarAirHockeySingle):
    def __init__(self, return_cost=True, dynamic_noise=0, headless=True, learning_constr=[]):
        self.return_cost = return_cost

        # if headless:
        super().__init__(horizon=300, viewer_params={'headless': headless, 'camera_params': {
                            'static': dict(distance=5.0, elevation=-45.0, azimuth=90.0, lookat=np.array([0.0, 0.0, 0.0])),
                            'follow': dict(distance=3.5, elevation=0.0, azimuth=90.0),
                            'top_static': dict(distance=5.0, elevation=-90.0, azimuth=90.0, lookat=np.array([0.0, 0.0, 0.0]))
                        }})

        # Compute Link Constraint Bound
        x_l = - self.env_info['robot']['base_frame'][0][0, 3] - (
                self.env_info['table']['length'] / 2 - self.env_info['mallet']['radius'])
        y_l = - (self.env_info['table']['width'] / 2 - self.env_info['mallet']['radius'])
        y_u = self.env_info['table']['width'] / 2 - self.env_info['mallet']['radius']

        self.link_constr_ub = np.array([x_l, y_l, -y_u])

        self.constr_dim = 1

        self.ee_puck_dist = np.inf

        self.dynamic_noise = dynamic_noise

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
        new_action = action.copy()
        if self.dynamic_noise > 0:
            new_action += np.random.normal(0, self.dynamic_noise, size=action.shape)
            new_action -= self.dynamic_noise * self.get_joints(self._obs.copy())[1]

        obs, reward, done, info = super(PlanarAirHockey, self).step(new_action)

        cost = info["cost"]

        if self.return_cost:
            return obs, reward, cost, done, info
        return obs, reward, done, info

    def setup(self, obs=None):
        puck_pos = np.random.uniform([-0.7, -0.4, -np.pi], [-0.3, 0.4, np.pi])
        self._write_data("puck_x_pos", puck_pos[0])
        self._write_data("puck_y_pos", puck_pos[1])
        self._write_data("puck_yaw_pos", puck_pos[2])

        super(PlanarAirHockey, self).setup(obs)

        self.absorb_type = AbsorbType.NONE
        self.ee_puck_dist = np.inf
        self.dist_to_goal = np.inf

        # self._debug_mallet_pos = []

    def reward(self, state, action, next_state, absorbing):
        # self._debug_mallet_pos.append(self.get_ee()[0][:2].copy())

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
            dist_to_goal = np.linalg.norm(np.array([2.43, 0]) - puck_pos)

            # if self.ee_puck_dist == np.inf:
            #     self.ee_puck_dist = ee_puck_dist
            # elif ee_puck_dist < self.ee_puck_dist:
            #     r += (self.ee_puck_dist - ee_puck_dist) * 50
            #     self.ee_puck_dist = ee_puck_dist

            if self.dist_to_goal == np.inf:
                self.dist_to_goal = dist_to_goal
            # elif dist_to_goal < self.dist_to_goal:
            r += (self.dist_to_goal - dist_to_goal) * 50
            self.dist_to_goal = dist_to_goal
            # if puck_pos[0] > 1.51:
            # r += 2 * np.clip(puck_vel[0], 0, 3)
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
        ee_pos = self._data.body("planar_robot_1/body_ee").xpos - pos_offset

        link_max = (np.array([-ee_pos[0], -ee_pos[1], ee_pos[1]]) + self.link_constr_ub).max()

        success = False
        if self.absorb_type == AbsorbType.GOAL:
            success = True

        return {'cost': max(q_max, link_max), 'success': success, 'q_cost': q_max, 'link_cost': link_max,
                'puck_vel': np.linalg.norm(puck_vel), 'joint_vel': np.linalg.norm(dq)}

    def _modify_observation(self, obs):
        obs = super()._modify_observation(obs)
        new_obs = obs.copy()
        puck_pos, puck_vel = self.get_puck(obs)
        ee_pos, ee_vel = self.get_ee()
        ee_pos[0] += 1.51
        ee_puck_pos = ee_pos[:2] - puck_pos[:2]
        ee_puck_vel = ee_vel[3:5] - puck_vel[:2]

        return np.concatenate([new_obs, ee_puck_pos, ee_puck_vel])

    def _modify_mdp_info(self, mdp_info):
        mdp_info = super()._modify_mdp_info(mdp_info)
        mdp_info.observation_space = Box(high=np.concatenate([mdp_info.observation_space.high, [2, 1, 5, 5]]),
                                         low=np.concatenate([mdp_info.observation_space.high, [-2, -1, 5, 5]]))
        return mdp_info

    def constraint_func(self, q):
        N = len(q)
        n_constr = self.original_constraint_list.output_dim()
        cons = np.zeros((N, n_constr))
        J_q = np.zeros((N, n_constr, q.shape[-1]))
        if n_constr != 0:
            for i in range(N):
                c, J = self._original_constraint(q[i])

                cons[i] = c
                J_q[i] = J[..., :q.shape[-1]]

        return cons, J_q, np.zeros((N, n_constr, 0)), self.K

    def _original_constraint(self, q):
        pos = q[:3]
        vel = q[3:] if len(q) > 3 else np.zeros(3)

        constraint_keys = self.original_constraint_list.keys()
        constraints = []
        constraints_J = []

        for key in constraint_keys:
            constraints.append(self.original_constraint_list.get(key).fun(pos, vel))
            constraints_J.append(self.original_constraint_list.get(key).jacobian(pos, vel).copy())

        const = np.concatenate(constraints)
        J_q = np.vstack(constraints_J)

        return const, J_q

class PlanarAirHockeyVel(VelocityControl, PlanarAirHockey):
    def __init__(self, return_cost=True, dynamic_noise=0, headless=True):
        p_gain = [1500., 1000., 500.] # 500 100 20            # 1000 500 100              # 1500 1000 500  # 500 500 500
        d_gain = [50., 10., 1.]                   # 10 2 1                  # 50 30 10                   # 50 50 50
        i_gain = [0, 0, 0]

        PlanarAirHockey.__init__(self, return_cost, dynamic_noise, headless)
        VelocityControl.__init__(self, p_gain=p_gain, d_gain=d_gain, i_gain=i_gain)

        constraints_class = [JointPositionConstraint, EndEffectorConstraint]

        K_values = [1.0, 0.5]
        
        self.constraint_init(constraints_class, K_values)
class PlanarAirHockeyAcc(AccelerationControl, PlanarAirHockey):
    def __init__(self, return_cost=True, dynamic_noise=0, headless=True):
        super(PlanarAirHockeyAcc, self).__init__(return_cost, dynamic_noise, headless)

        constraints_class = [JointPositionConstraint, EndEffectorConstraint, JointVelocityConstraint]

        K_values = [1.0, 0.5, 0.]

        self.constraint_init(constraints_class, K_values)

    def _preprocess_action(self, action):
        action = super(PlanarAirHockeyAcc, self)._preprocess_action(action)
        return action * self.env_info['robot']['joint_acc_limit'][1]

    def _create_info_dictionary(self, state):
        q, dq = self.get_joints(state)

        dq_max = np.concatenate([-dq + self.env_info['robot']['joint_vel_limit'][0] * 0.95,
                                 dq - self.env_info['robot']['joint_vel_limit'][1] * 0.95]).max()

        info = super(PlanarAirHockeyAcc, self)._create_info_dictionary(state)

        info['cost'] = max(dq_max, info['cost'])

        info.update({'dq_cost': dq_max})

        return info

if __name__ == '__main__':
    env = PlanarAirHockeyAcc()
    env.reset()
    env.render()
    while True:
        action = np.array([0, 0, 1])
        s, r, c, done, info = env.step(action)

        # env.cost(np.tile(s[None, :], (10, 1)))

        env.render()

        if done:
            env.reset()
