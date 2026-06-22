import numpy as np

from cremini_rl.constraints.constraints import ConstraintCollection as ConstraintList

class ConstrEnv:
    def constraint_init(self, constraints_class, K_values):
        self.K = []
        self.original_constraint_list = ConstraintList()

        for constr_class, k in zip(constraints_class, K_values):
            constr = constr_class(self.env_info)
            self.original_constraint_list.add(constr)
            self.K += [k] * constr.output_dim

        self.K = np.array(self.K)
        self.dim_q = self.info.action_space.shape[-1]

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
        pos = q[:self.dim_q]
        vel = q[self.dim_q:] if len(q) > self.dim_q else np.zeros(self.dim_q)

        constraint_keys = self.original_constraint_list.keys()
        constraints = []
        constraints_J = []

        for key in constraint_keys:
            constraints.append(self.original_constraint_list.get(key).fun(pos, vel))
            constraints_J.append(self.original_constraint_list.get(key).jacobian(pos, vel).copy())

        const = np.concatenate(constraints)
        J_q = np.vstack(constraints_J)

        return const, J_q