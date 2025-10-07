import numpy as np
import mujoco
from mushroom_rl.utils.spaces import Box

class VelocityControl:
    def _compute_vel_action(self, obs, action):
        q, dq = self.get_joints(obs)
        vel_high = np.minimum(self.env_info['robot']['joint_vel_limit'][1],
                              5 * (self.env_info['robot']['joint_pos_limit'][1] - q))
        vel_low = np.maximum(self.env_info['robot']['joint_vel_limit'][0],
                             5 * (self.env_info['robot']['joint_pos_limit'][0] - q))
        vel = np.clip(action, vel_low, vel_high)
        self.env_info['robot']['robot_data'].qpos[:] = q
        self.env_info['robot']['robot_data'].qvel[:] = vel
        torque = np.zeros(self.env_info['robot']['n_joints'])
        mujoco.mj_inverse(self.env_info['robot']['robot_model'], self.env_info['robot']['robot_data'])
        torque = self.env_info['robot']['robot_data'].qfrc_inverse
        return torque


class AccelerationControl:
    def _compute_acc_action(self, obs, action):
        q, dq = self.get_joints(obs)
        acc_high = np.minimum(self.env_info['robot']['joint_acc_limit'][1],
                              5 * (self.env_info['robot']['joint_vel_limit'][1] - dq))
        acc_low = np.maximum(self.env_info['robot']['joint_acc_limit'][0],
                             5 * (self.env_info['robot']['joint_vel_limit'][0] - dq))
        acc = np.clip(action, acc_low, acc_high)
        self.env_info['robot']['robot_data'].qpos[:] = q
        self.env_info['robot']['robot_data'].qvel[:] = dq
        self.env_info['robot']['robot_data'].qacc[:] = acc
        torque = np.zeros(self.env_info['robot']['n_joints'])
        mujoco.mj_inverse(self.env_info['robot']['robot_model'], self.env_info['robot']['robot_data'])
        torque = self.env_info['robot']['robot_data'].qfrc_inverse
        return torque

    def _modify_mdp_info(self, mdp_info):
        super(AccelerationControl, self)._modify_mdp_info(mdp_info)
        mdp_info.action_space = Box(low=-np.ones(self.env_info['robot']['n_joints']),
                                    high=np.ones(self.env_info['robot']['n_joints']))
        return mdp_info