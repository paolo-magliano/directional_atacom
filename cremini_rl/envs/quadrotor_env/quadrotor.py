import os, uuid
import mujoco
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

from mushroom_rl.environments import MuJoCo
from mushroom_rl.environments.mujoco import MuJoCo, ObservationType
from mushroom_rl.utils.spaces import Box

class QuadrotorNavigation(MuJoCo):
    def __init__(self, gamma=0.995, horizon=2000, timestep=0.01, constr_margin=0., collision=False, n_substeps=1, n_intermediate_steps=1, disturbance=False, viewer_params={}):

        init_height = 0.5
        bounds = np.array([-0.8, 0.8, -0.6, 0.6, init_height - 0.4, init_height + 0.5])
        points = np.array([[0, 0, 0.3]])

        xml_path = self._setup_task(horizon, bounds, points, init_height, collision, constr_margin)
 
        actuation_spec = ['body_thrust', 'x_moment', 'y_moment', 'z_moment']

        observation_spec = [("pos", "cf2", ObservationType.BODY_POS),
                            ("rot", "cf2", ObservationType.BODY_ROT),
                            ("vel", "cf2", ObservationType.BODY_VEL)]

        if viewer_params == {}:
            viewer_params = {'camera_params': {'static': dict(
                                                    distance=3.5,
                                                    elevation=-45.0,  
                                                    azimuth=90.0,     
                                                    lookat=(0., 0.7, 0.)),
                                               'follow': dict(distance=3.0, elevation=-45.0, azimuth=90.0)},
                             'width': 1920, 'height': 1080,
                             'default_camera_mode': 'static',
                             'hide_menu_on_startup': True,
                             'headless': True}

        self.disturbance = disturbance

        super().__init__(str(xml_path), actuation_spec, observation_spec, gamma, horizon, timestep=timestep,
                         n_substeps=n_substeps, n_intermediate_steps=n_intermediate_steps, additional_data_spec=None,
                         collision_groups=None, max_joint_vel=None, **viewer_params)
        
        xml_path.unlink(missing_ok=True)

        self.env_info = {
            'robot': {'n_joints': 7},
            'mass': self._model.body('cf2').mass,
            'inertia': self._model.body('cf2').inertia,
            'gear_ratios': np.array([self._model.actuator('x_moment').gear[-3],
                                     self._model.actuator('y_moment').gear[-2],
                                     self._model.actuator('z_moment').gear[-1]]),
            'gravity': -self._model.opt.gravity[2],
            'dt': self.dt,
            'bounds': bounds,
            'points': points,
            'constr_margin': constr_margin}

    def _setup_task(self, horizon, bounds, points, init_height, collision, constr_margin):
        self._init_height = init_height
        self.des_traj = desired_trajectory(horizon, self._init_height)
        self.des_vel_traj = desired_vel_trajectory(horizon, self._init_height)
        self._target_idx = 0

        return self._setup_xml_world(bounds, points, init_height, collision, constr_margin)

    def setup(self, obs):
        x_init, y_init = 0., 0.
        for px, py, pr in self.env_info['points']:
            if np.linalg.norm(np.array([x_init, y_init]) - np.array([px, py])) < pr:
                x_init = px + pr + 0.1
        
        self._data.qpos = np.array([x_init, y_init, self._init_height, 0., 0., 0., 0.])
        mujoco.mj_kinematics(self._model, self._data)

        return super().setup(obs)

    @property
    def target_point(self):
        return self.des_traj[self._target_idx]
    
    @property
    def target_velocity(self):
        return self.des_vel_traj[self._target_idx]

    def get_pos(self, state):
        return state[..., :3]
    
    def get_vel(self, state):
        return state[..., 7:10]
    
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
        dist_vel = np.linalg.norm(self.target_velocity - state[7:10])
        omega_z = np.linalg.norm(state[12])
        reward = np.exp(-5 * dist ** 2) + 0.1 * np.exp(-5 * dist_vel ** 2)
        return reward

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

        success = False
        if self.absorb_type == AbsorbType.GOAL:
            success = True

        return {'cost': np.concatenate([q_constr, link_constr]), 'success': success,  'q_cost': q_constr.max(), 'link_cost': link_constr.max(),
                'puck_vel': np.linalg.norm(puck_vel), 'joint_vel': np.linalg.norm(dq)}

    def _modify_observation(self, obs):
        # Modify the observation to the North-East-Down frame
        obs = super()._modify_observation(obs)

        body_vel_rot = self._data.sensor('body_gyro').data

        obs[7:10] = self._data.body('cf2').cvel[3:6]
        obs[10:13] = body_vel_rot

        obs = np.concatenate([obs, self.target_point, self.target_velocity])
        return obs

    def _modify_mdp_info(self, mdp_info):
        obs_low = np.array([-10., -10, 0., -1., -1., -1., -1., -10., -10., -10., -10., -10., -10., -3., -3., 0., -10., -10., -10.])
        obs_high = np.array([10., 10., 10, 1., 1., 1., 1., 10., 10., 10., 10., 10., 10., 3., 3., 1., 10., 10., 10.])
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

    def _setup_xml_world(self, bounds, points, init_height, collision, constr_margin):
        tree = ET.parse(os.path.dirname(__file__) + "/cf2_safety_base.xml")
        root = tree.getroot()
        worldbody = root.find("worldbody")

        def element(worldbody, element_type, name, shape, size, x, y, z):
            return  ET.SubElement(worldbody, element_type, {
                        "name": name,
                        "type": shape,
                        "size": size,
                        "pos": f"{x} {y} {z}",
                        "material": "wall",
                    })

        def point_element(worldbody, element_type, name, shape, x, y, z, r):
            return element(worldbody, element_type, name, shape, f"{r} {z}", x, y, z)
        
        def bound_element(worldbody, element_type, name, shape, x, y, z, sx, sy):
            return element(worldbody, element_type, name, shape, f"{sx} {sy} {z}", x, y, z)

 
        for i, (x, y, r) in enumerate(points, start=1):
            if collision:
                point_element(worldbody, "geom", f"point_geom_{i}", "cylinder", x, y, self._init_height, r)
                if constr_margin > 0:
                    point_element(worldbody, "site", f"point_{i}", "cylinder", x, y, self._init_height, r + constr_margin)
            else:
                point_element(worldbody, "site", f"point_{i}", "cylinder", x, y, self._init_height, r)


        wall_thickness = 0.01

        x_center = 0.5 * (bounds[0] + bounds[1])
        y_center = 0.5 * (bounds[2] + bounds[3])
        x_size = np.abs(bounds[3] - bounds[2]) / 2 + wall_thickness / 2
        y_size = np.abs(bounds[1] - bounds[0]) / 2 + wall_thickness / 2

        walls_config = [
            ("north", x_center, bounds[3], y_size, wall_thickness, 0, - constr_margin), 
            ("south", x_center, bounds[2], y_size, wall_thickness, 0, + constr_margin),
            ("east",  bounds[1], y_center, wall_thickness, x_size, - constr_margin, 0),
            ("west",  bounds[0], y_center, wall_thickness, x_size, + constr_margin, 0),
        ]

        for suffix, wx, wy, sx, sy, mx, my in walls_config:
            if collision:
                bound_element(worldbody, "geom", f"wall_{suffix}_geom", "box", wx, wy, self._init_height, sx, sy)
                if constr_margin > 0:
                    bound_element(worldbody, "site", f"wall_{suffix}", "box", wx + mx, wy + my, self._init_height, sx - np.abs(my), sy - np.abs(mx))
            else:
                bound_element(worldbody, "site", f"wall_{suffix}", "box", wx, wy, self._init_height, sx, sy)

        # Init point
        ET.SubElement(worldbody, "site", {
            "name": "target",
            "type": "sphere",
            "size": "0.02",
            "pos": f"0 0 {self._init_height}",
            "rgba": "0 0.8 0 1"
        })

        unique = f"{os.getpid()}_{uuid.uuid4().hex}"
        write_name = f"cf2_safety_obstacles_{unique}.xml"

        xml_path = Path(os.path.dirname(__file__)) / write_name
        xml_path.parent.mkdir(parents=True, exist_ok=True)

        tree.write(xml_path, encoding="utf-8", xml_declaration=True)

        return xml_path

def desired_trajectory(N_points, height, base=1.):
    # Generate a 8-shaped trajectory
    t = np.linspace(0, 2 * np.pi, N_points)
    x = base * np.sin(t)
    y = base * np.sin(t) * np.cos(t)
    traj = np.vstack([x, y, np.ones_like(x) * height]).T
    return traj

def desired_vel_trajectory(N_points, height, base=1.):
    # Generate a 8-shaped trajectory
    t = np.linspace(0, 2 * np.pi, N_points)
    dx = base * np.cos(t)
    dy = base * (np.cos(t)**2 - np.sin(t)**2)
    traj = np.vstack([dx, dy, np.zeros_like(dx)]).T
    return traj