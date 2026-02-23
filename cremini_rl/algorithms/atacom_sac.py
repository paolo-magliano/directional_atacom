import torch
from mushroom_rl.algorithms.actor_critic.deep_actor_critic import SAC
from mushroom_rl.utils.parameters import to_parameter

from cremini_rl.utils.null_space import batch_smooth_basis, smooth_basis
import numpy as np

from scipy.linalg import qr, svd

class AtacomSACBaseline(SAC):
    def __init__(self, mdp_info, control_system, atacom_lam, atacom_beta, atacom_dc, constraint_func, use_viability,
                 actor_mu_params,
                 actor_sigma_params, actor_optimizer, critic_params, batch_size,
                 initial_replay_size, max_replay_size, warmup_transitions, tau, lr_alpha, use_log_alpha_loss=False,
                 log_std_min=-20, log_std_max=2, target_entropy=None, critic_fit_params=None):
        super().__init__(mdp_info, actor_mu_params, actor_sigma_params, actor_optimizer, critic_params, batch_size,
                         initial_replay_size, max_replay_size, warmup_transitions, tau, lr_alpha, use_log_alpha_loss,
                         log_std_min, log_std_max, target_entropy, critic_fit_params)

        self._control_system = control_system
        self._atacom_lam = to_parameter(atacom_lam)
        self._atacom_beta = to_parameter(atacom_beta)
        self._constraint_func = constraint_func

        self._add_save_attr(_control_system='mushroom',
                            _atacom_lam='mushroom',
                            _atacom_beta='mushroom')

        self._use_viability = use_viability
        self._atacom_dc = atacom_dc
        self.derivation_step_size = 1e-4
        # self.K = 0.5


    def fit(self, dataset, **info):
        new_dataset = []
        for sample in dataset:
            new_dataset.append(sample[:4] + sample[5:])

        super().fit(new_dataset, **info)

    def J_slack(self, slack):
        out = np.zeros(slack.shape + (slack.shape[1],))
        np.einsum('ijj->ij', out)[:] = 1 / np.maximum(np.exp(-self._atacom_beta() * slack), 1e-5) - 1

        return out

    def preprocess_action(self, state, alpha, next_cost):
        alpha = np.clip(alpha, self.mdp_info.action_space.low, self.mdp_info.action_space.high)

        state_tensor = torch.from_numpy(state)
        q = self._control_system.get_q(state_tensor.numpy())
        
        # Only get classic constraint k(s), need to implement viability here
        cons, J_k, J_x, K = self._constraint_func(q)
        lam = self._atacom_lam()

        G = self._control_system.G(q)
        f = self._control_system.f(q)

        drift = np.zeros(cons.shape + (1,))

        if self._use_viability:
            # Assumes states is stacked as [q, q_dot]
            state_dim = G.shape[0] // 2
            q_dot = q[:, state_dim:, None]

            # use lower half of the system as new dynamics
            G = G[state_dim:]
            f = f[:, state_dim:]

            indicator = K == 0
            # Only velocity jacobian for vel constraint
            J_k_velocity = J_k[:, indicator, state_dim:]
            # Only position jacobian for viability constraint
            J_k_viability = J_k[:, ~indicator, :state_dim]

            # Build new jacobian
            J_k = np.concatenate([J_k_viability, J_k_velocity], axis=1)

            # Transform to viability constraint
            cons = cons + K * (J_k @ q_dot).squeeze(2)

            q_delta = q.copy()
            q_delta[:, :state_dim] += q_delta[:, state_dim:] * self.derivation_step_size
            _, J_k_delta, _, _ = self._constraint_func(q_delta)
            J_k_viability_delta = J_k_delta[:, ~indicator, :state_dim]
            J_k_dot = (J_k_viability_delta - J_k_viability) / self.derivation_step_size

            drift[:, ~indicator] += (J_k_viability + J_k_dot) @ q_dot

            # Estimate J_k_dot by finite difference
            # J_k_dot = np.zeros(J_k_viability.shape + (state_dim,))
            # for i in range(state_dim):
            #     q_delta = q.copy()
            #     q_delta[:, i] += self.derivation_step_size
            #     _, J_k_delta, _, _ = self._constraint_func(q_delta)
            #     J_k_viability_delta = J_k_delta[:, ~indicator, :state_dim]
            #     J_k_dot[:, :, :, i] = (J_k_viability_delta - J_k_viability) / self.derivation_step_size

            # drift[:, ~indicator] += (J_k_viability + (J_k_dot @ q_dot).squeeze(3)) @ q_dot

        slack = np.maximum(-cons, 1e-5)

        J_G = J_k @ G  # (B, k, u)

        drift += J_k @ f  # (B, k)

        drift = np.maximum(drift, 0).squeeze(-1)

        c = cons + slack

        J_u = np.concatenate((J_G, self.J_slack(slack)), axis=-1)  # (B, k, u + k)

            # J_u[np.logical_not(useful_constr)] = 0
            # while sum(J_u[0]) == 0:
            #     J_u = J_u[1:]
            #     drift = drift[1:]
            #     c = c[1:]

        # Discard non-useful constraints thanks removing lines
        # J_u = J_u[:, useful_constr.squeeze()][..., np.concatenate((np.ones((alpha.shape[-1]), dtype=bool), useful_constr.squeeze()))]
        # drift = drift[:, useful_constr.squeeze()]
        # c = c[:, useful_constr.squeeze()]
            
            # B_u = batch_smooth_basis(J_u[None, :])[..., :alpha.shape[-1]].squeeze(0)

        b = np.concatenate([np.linalg.lstsq(J_u[i], - drift[i] - lam * c[i], rcond=None)[0] for i in range(J_u.shape[0])], axis=0)

        # Discard non-useful constraints thanks to slack variables
        useful_constr = self._directional_constraints(drift, J_G, alpha)
        slack[np.logical_not(useful_constr)] = 1e+2
        J_u_dir = np.concatenate((J_G, self.J_slack(slack)), axis=-1) # J_u[:, useful_constr.squeeze()][..., np.concatenate((np.ones((alpha.shape[-1]), dtype=bool), useful_constr.squeeze()))]

        B_u = batch_smooth_basis(J_u_dir)[..., :alpha.shape[-1]]

        tangential_term = (B_u @ alpha[:, None]).squeeze(-1)
        action = tangential_term[..., :alpha.shape[-1]] + b[..., :alpha.shape[-1]]

        return action

    def _directional_constraints(self, drift, J_G, alpha):
        if self._atacom_dc:
            constraint_direction = J_G @ alpha
            useful_constr = constraint_direction > 0
        else:
            useful_constr = np.ones(drift.shape, dtype=bool)
        return useful_constr