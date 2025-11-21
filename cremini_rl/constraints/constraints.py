import numpy as np

from air_hockey_challenge.constraints import ConstraintList, Constraint

class ConstraintBase(Constraint):
    def __init__(self, env_info, output_dim, n_joints=None, **kwargs):
        self._env_info = env_info
        self._name = None

        self.output_dim = output_dim
        self.n_joints = n_joints if n_joints is not None else 2 * env_info["robot"]["n_joints"]

        self._fun_value = np.zeros(self.output_dim)
        self._jac_value = np.zeros((self.output_dim, self.n_joints))
        self._q_prev = None
        self._dq_prev = None

    def fun(self, q, z=None) -> np.ndarray:
        pass

    def jacobian(self, q, z=None) -> np.ndarray:
        pass

class ConstraintCollection(ConstraintList):
    def output_dim(self):
        return sum(constr.output_dim for constr in self.constraints.values())
