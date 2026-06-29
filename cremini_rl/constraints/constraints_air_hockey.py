from air_hockey_challenge.constraints import EndEffectorConstraint as EEConstraint
from air_hockey_challenge.constraints import LinkConstraint as LConstraint

class EndEffectorConstraint(EEConstraint):
    def __init__(self, env_info, **kwargs):
        super().__init__(env_info, **kwargs)
        tolerance = 0.02
        z_tolerance = 0.005

        self.x_lb = - self._env_info['robot']['base_frame'][0][0, 3] - (
                self._env_info['table']['length'] / 2 - self._env_info['mallet']['radius']) + tolerance
        self.y_lb = - (self._env_info['table']['width'] / 2 - self._env_info['mallet']['radius']) + tolerance
        self.y_ub = (self._env_info['table']['width'] / 2 - self._env_info['mallet']['radius']) + tolerance
        self.z_lb = self._env_info['robot']['ee_desired_height'] - z_tolerance
        self.z_ub = self._env_info['robot']['ee_desired_height'] + z_tolerance

class LinkConstraint(LConstraint):
    def __init__(self, env_info, **kwargs):
        super().__init__(env_info, **kwargs)
        tolerance = 0.02

        self.z_lb = 0.35 + tolerance
        