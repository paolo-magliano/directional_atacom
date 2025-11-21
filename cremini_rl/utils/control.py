import numpy as np
import mujoco
from mushroom_rl.utils.spaces import Box
from air_hockey_challenge.environments.position_control_wrapper import PositionControl

class VelocityControl(PositionControl):
    def __init__(self, p_gain, d_gain, i_gain, interpolation_order=-1, debug=False, *args, **kwargs):
        super(VelocityControl, self).__init__(p_gain, d_gain, i_gain, interpolation_order, debug, *args, **kwargs)
        self._pos = np.zeros(self.env_info['robot']['n_joints'])
        self._vel = np.zeros(self.env_info['robot']['n_joints'])

    def _preprocess_action(self, action):
        return action

    def _step_init(self, obs, action):
        self._pos, self._vel = self._integrator(self.env_info['robot']['joint_vel_limit'][1] * action, obs)
        super(VelocityControl, self)._step_init(obs, np.vstack((self._pos, self._vel)))

    def _compute_action(self, obs, action):
        return super(VelocityControl, self)._compute_action(obs, np.vstack((self._pos, self._vel)))

    def _integrator(self, integrand, obs):
        """
        We first convert the state to the original state and get actual position and velocity.
        """
        pos, vel = self.get_joints(obs)

        # Compute the soft limit of the acceleration,
        # details can be found here: http://wiki.ros.org/pr2_controller_manager/safety_limits
        vel_soft_limit = np.clip(-5 * (pos - self.env_info['robot']['joint_pos_limit']), self.env_info['robot']['joint_vel_limit'][0], self.env_info['robot']['joint_vel_limit'][1])

        clipped_vel = np.clip(integrand, *vel_soft_limit).squeeze()
        pos += clipped_vel * self.dt
        vel = clipped_vel.copy()

        return pos, vel

class VelocityControlIIWA(VelocityControl):
    def __init__(self, *args, **kwargs):
        p_gain = [1500., 1500., 1200., 1200., 1000., 1000., 500.]
        d_gain = [60, 80, 60, 30, 10, 1, 0.5]
        i_gain = [0, 0, 0, 0, 0, 0, 0]

        super(VelocityControlIIWA, self).__init__(p_gain=p_gain, d_gain=d_gain, i_gain=i_gain, *args, **kwargs)

class AccelerationControl:
    def _compute_action(self, obs, action):
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