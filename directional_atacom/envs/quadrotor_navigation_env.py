import numpy as np

from directional_atacom.envs.base_constr_env import ConstrEnv
from directional_atacom.envs.quadrotor_env.quadrotor import QuadrotorNavigation

from directional_atacom.constraints.constraints_quadrotor import ZPositionConstraint, ZRotationConstraint, WallConstraint, XYPointsConstraint

class QuadrotorEnv(QuadrotorNavigation, ConstrEnv):
    def __init__(self, return_cost=True, gamma=0.995, horizon=2000, timestep=0.01):
        super().__init__(gamma=gamma, horizon=horizon, timestep=timestep)
        self.return_cost = return_cost

        constraints_class = [ZPositionConstraint, ZRotationConstraint, WallConstraint, XYPointsConstraint]

        K_values = [0.5, 0., 0.5, 0.5]
        
        self.constraint_init(constraints_class, K_values)
        self.dim_q = self.env_info['robot']['n_joints']

    def step(self, action):
        obs, reward, done, info = super(QuadrotorEnv, self).step(action)

        cost = info["cost"]

        if self.return_cost:
            return obs, reward, cost, done, info
        return obs, reward, done, info
    
    def _create_info_dictionary(self, state):
        pos = self.get_pos(state)
        vel = self.get_vel(state)

        points_constr = np.array([-np.linalg.norm(pos[:2] - np.array([x, y])) + r for x, y, r in self.env_info['points']])

        bounds_constr = np.array([self.env_info['bounds'][0], -self.env_info['bounds'][1], \
                                self.env_info['bounds'][2], -self.env_info['bounds'][3], \
                                self.env_info['bounds'][4], -self.env_info['bounds'][5]]) + \
                                np.array([-pos[0], pos[0], -pos[1], pos[1], -pos[2], pos[2]])

        dist_to_target = np.linalg.norm(self.target_point - pos)
        dist_to_target_vel = np.linalg.norm(self.target_velocity - vel)

        return {'cost': np.concatenate([points_constr, bounds_constr]), 'points_constr': points_constr.max(), 'bounds_constr': bounds_constr.max(),
                'dist_to_target': dist_to_target, 'dist_to_target_vel': dist_to_target_vel}


    
