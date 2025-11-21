import os
import mujoco
import numpy as np

import xml.etree.ElementTree as ET
from pathlib import Path

from mushroom_rl.environments import MuJoCo
from mushroom_rl.environments.mujoco import MuJoCo, ObservationType
from mushroom_rl.utils.spaces import Box
from mushroom_rl.utils.dataset import parse_dataset

from cremini_rl.constraints.constraints_quadrotor import *
from cremini_rl.constraints.constraints import ConstraintCollection as ConstraintList

assets_path = os.path.dirname(__file__) + "/quadrotor/"

def generate_desired_trajectory(N_points, height, base=1.):
    # Generate a 8-shaped trajectory
    t = np.linspace(0, 2 * np.pi, N_points)
    x = base * np.sin(t)
    y = base * np.sin(t) * np.cos(t)
    traj = np.vstack([x, y, np.ones_like(x) * height]).T
    return traj


class QuadrotorBase(MuJoCo):
    @classmethod
    def build_env(cls, **kwargs):
        return QuadrotorBase(**kwargs)

    def __init__(self, bound, points, point_radius=0.1, base=1., gamma=0.995, horizon=2000, timestep=0.01,
                 n_substeps=1, n_intermediate_steps=1, disturbance=True, viewer_params={}):

        tree = ET.parse(assets_path + "cf2_safety_base.xml")
        root = tree.getroot()
        worldbody = root.find("worldbody")

        x_center = 0.5 * (bound[0] + bound[1])
        y_center = 0.5 * (bound[2] + bound[3])
        x_size = np.abs(bound[3] - bound[2]) / 2
        y_size = np.abs(bound[1] - bound[0]) / 2

        self._init_height = 0.5
        self.des_traj = generate_desired_trajectory(horizon, self._init_height, base)
        self._target_idx = 0

        for i, (x, y) in enumerate(points, start=1):
            geom = ET.SubElement(worldbody, "site", {
                "name": f"pillar_{i}",
                "type": "cylinder",
                "size": f"{point_radius} {self._init_height}",
                "pos": f"{x} {y} {self._init_height}",
                "material": "wall",
                # "contype": "1",
                # "conaffinity": "1",
            })

        # North ( +y )
        ET.SubElement(worldbody, "site", {
            "name": "wall_north",
            "type": "box",
            "size": f"{y_size} {0.01} {self._init_height}",
            "pos":  f"{x_center} {bound[3]} {self._init_height}",
            "material": "wall"
        })
        # South ( -y )
        ET.SubElement(worldbody, "site", {
            "name": "wall_south",
            "type": "box",
            "size": f"{y_size} {0.01} {self._init_height}",
            "pos":  f"{x_center} {bound[2]} {self._init_height}",
            "material": "wall"
        })
        # East ( +x )
        ET.SubElement(worldbody, "site", {
            "name": "wall_east",
            "type": "box",
            "size": f"{0.01} {x_size} {self._init_height}",
            "pos":  f"{bound[1]} {y_center} {self._init_height}",
            "material": "wall"
        })
        # West ( -x )
        ET.SubElement(worldbody, "site", {
            "name": "wall_west",
            "type": "box",
            "size": f"{0.01} {x_size} {self._init_height}",
            "pos":  f"{bound[0]} {y_center} {self._init_height}",
            "material": "wall"
        })

        # Init point
        ET.SubElement(worldbody, "site", {
            "name": "target",
            "type": "sphere",
            "size": "0.02",
            "pos": f"0 0 {self._init_height}",
            "rgba": "0 0.8 0 1"
        })

        # Write out the modified XML
        Path(assets_path + "cf2_safety_obstacles.xml").parent.mkdir(parents=True, exist_ok=True)
        tree.write(assets_path + "cf2_safety_obstacles.xml", encoding="utf-8", xml_declaration=True)

        xml_file = assets_path + "cf2_safety_obstacles.xml"

        actuation_spec = ['body_thrust', 'x_moment', 'y_moment', 'z_moment']

        observation_spec = [("pos", "cf2", ObservationType.BODY_POS),
                            ("rot", "cf2", ObservationType.BODY_ROT),
                            ("vel", "cf2", ObservationType.BODY_VEL)]

        if viewer_params == {}:
            viewer_params = {'camera_params': {'static': dict(distance=3.0, elevation=-30.0, azimuth=90.0,
                                                              lookat=(0., 0., 0.)),
                                               'follow': dict(distance=3.0, elevation=-45.0, azimuth=90.0)},
                             'width': 1440, 'height': 810,
                             'default_camera_mode': 'follow',
                             'hide_menu_on_startup': True,
                             'headless': True}

        self.disturbance = disturbance

        super().__init__(xml_file, actuation_spec, observation_spec, gamma, horizon, timestep=timestep,
                         n_substeps=n_substeps, n_intermediate_steps=n_intermediate_steps, additional_data_spec=None,
                         collision_groups=None, max_joint_vel=None, **viewer_params)

        self.env_info = {
            'mass': self._model.body('cf2').mass,
            'inertia': self._model.body('cf2').inertia,
            'gear_ratios': np.array([self._model.actuator('x_moment').gear[-3],
                                     self._model.actuator('y_moment').gear[-2],
                                     self._model.actuator('z_moment').gear[-1]]),
            'gravity': -self._model.opt.gravity[2],
            'dt': self.dt,
            'bound': bound,
            'points': points,
            'point_radius': point_radius}

    def setup(self, obs):
        self._data.qpos = np.array([-0., 0., self._init_height, 0., 0., 0., 0.])
        mujoco.mj_kinematics(self._model, self._data)

        return super().setup(obs)

    @property
    def target_point(self):
        return self.des_traj[self._target_idx]
    
    def _step_finalize(self):
        self._data.site('target').xpos = self.des_traj[self._target_idx]
        self._target_idx += 1
        if self._target_idx >= len(self.des_traj):
            self._target_idx = 0

    def is_absorbing(self, obs):
        if obs[2] <= 0.03 or np.abs(obs[1]) > 5 or np.abs(obs[0]) > 5:
            return True
        return False

    def reward(self, state, action, next_state, absorbing):
        dist = np.linalg.norm(self.target_point - state[:3])
        reward = np.exp(-10 * dist ** 2)
        return reward

    def _modify_observation(self, obs):
        # Modify the observation to the North-East-Down frame
        obs = super()._modify_observation(obs)

        body_vel_rot = self._data.sensor('body_gyro').data

        obs[7:10] = self._data.body('cf2').cvel[3:6]
        obs[10:13] = body_vel_rot

        obs = np.concatenate([obs, self.target_point])
        return obs

    def _modify_mdp_info(self, mdp_info):
        obs_low = np.array([-10., -10, 0., -1., -1., -1., -1., -10., -10., -10., -10., -10., -10., -3., -3., 0.])
        obs_high = np.array([10., 10., 10, 1., 1., 1., 1., 10., 10., 10., 10., 10., 10., 3., 3., 1.])
        mdp_info.observation_space = Box(obs_low, obs_high)
        return mdp_info

    def _compute_action(self, obs, action):
        # The action is the desired thrust and moments
        ctrl_action = action.copy()
        return ctrl_action

    def _preprocess_action(self, action):
        ctrl_action = np.asarray(action).flatten()[:4].copy()
        return ctrl_action

    def _simulation_pre_step(self):
        if self.disturbance:
            self._data.qfrc_applied[0:3] = np.random.normal(0, 0.02, 3)

class QuadrotorEnv(QuadrotorBase):
    def __init__(self, return_cost=True, **kwargs):
        base = 1.
        margin = 0.25
        points = np.array([[-1 * base, -0.5 * base], [-1 * base, 0.5 * base], [-0.5 * base, 0], [0, -0.5 * base], [0, 0.5 * base], [0.5 * base, 0], [1 * base, -0.5 * base], [1 * base, 0.5 * base]])
        bound = np.array([points[:,0].min() - margin, points[:,0].max() + margin,
                        points[:,1].min() - margin, points[:,1].max() + margin])
        points = np.array([[-0.5 * base, 0], [0.5 * base, 0]])
        point_radius = 0.05

        super().__init__(bound, points, point_radius, **kwargs)

        self.return_cost = return_cost

        self.original_constraint_list = ConstraintList()
        self.original_constraint_list.add(WallConstraint(self.env_info, bound))
        self.original_constraint_list.add(ZPositionConstraint(self.env_info, [-0.98, 0.1]))
        # self.original_constraint_list.add(XYPointsConstraint(self.env_info, points, point_radius))

    def step(self, action):
        obs, reward, done, info = super(QuadrotorBase, self).step(action)

        cost = info["cost"]

        if self.return_cost:
            return obs, reward, cost, done, info
        return obs, reward, done, info

    def _create_info_dictionary(self, state):
        pos = state[:3]
        vel = state[7:10]

        points_cost = np.array([-np.linalg.norm(pos[:2] - np.array([x, y])) + self.env_info['point_radius'] for x, y in self.env_info['points']])

        bound_cost = np.array([self.env_info['bound'][0], -self.env_info['bound'][1], \
                            self.env_info['bound'][2], -self.env_info['bound'][3]]) + \
                            np.array([-pos[0], pos[0], -pos[1], pos[1]])
        
        z_cost = np.array([0.1, -(1)]) + np.array([-pos[2], pos[2]])

        return {'cost': max(0, points_cost.max(), bound_cost.max(), z_cost.max()), 'points_cost': points_cost.max(), 'bound_cost': bound_cost.max(), 'z_cost': z_cost.max(),
                'dist_to_target': np.linalg.norm(self.target_point - pos),
                'cartesian_velocity': np.linalg.norm(vel)}

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

        return cons, J_q, np.zeros((N, n_constr, 0)), np.zeros((N, n_constr))

    def _original_constraint(self, q):

        constraint_keys = self.original_constraint_list.keys()
        constraints = []
        constraints_J = []

        for key in constraint_keys:
            constraints.append(self.original_constraint_list.get(key).fun(q))
            constraints_J.append(self.original_constraint_list.get(key).jacobian(q).copy())

        const = np.concatenate(constraints)
        J_q = np.vstack(constraints_J)

        return const, J_q

