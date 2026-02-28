import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

from mushroom_rl.algorithms.actor_critic.deep_actor_critic import DeepAC
from mushroom_rl.policy import Policy
from mushroom_rl.approximators import Regressor
from mushroom_rl.approximators.parametric import TorchApproximator
from mushroom_rl.utils.torch import to_float_tensor
from mushroom_rl.utils.parameters import to_parameter
from mushroom_rl.utils.replay_memory import ReplayMemory

from copy import deepcopy
from itertools import chain
from inspect import signature

from scipy.stats import norm

from cremini_rl.utils.null_space import batch_smooth_basis

from cremini_rl.utils.constraint_replay_memory import SafeReplayMemory

constr_aggregation_mapping = {
    'none': lambda e, **kwargs: e,
    'array': lambda e, **kwargs: e,
    'max': np.max
}

class DatacomSACPolicy(Policy):
    """
    Class used to implement the policy used by the Soft Actor-Critic algorithm.
    The policy is a Gaussian policy squashed by a tanh. This class implements the compute_action_and_log_prob and the
    compute_action_and_log_prob_t methods, that are fundamental for the internals calculations of the SAC algorithm.

    """

    def __init__(self, mu_approximator, sigma_approximator, constraint_value_function_approximator, constraint_approximator, control_system, mdp_info,
                 accepted_risk, delta, delta_value_function, atacom_lam, atacom_beta, target_entropy, min_a, max_a, log_std_min, log_std_max, atacom_dc, use_viability,
                 analytical_const=None):
        """
        Constructor.

        Args:
            mu_approximator (Regressor): a regressor computing mean in given a state;
            sigma_approximator (Regressor): a regressor computing the variance in given a state;
            min_a (np.ndarray): a vector specifying the minimum action value for each component;
            max_a (np.ndarray): a vector specifying the maximum action value for each component.
            log_std_min ([float, Parameter]): min value for the policy log std;
            log_std_max ([float, Parameter]): max value for the policy log std.

        """

        if mu_approximator.model.use_cuda:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self._mu_approximator = mu_approximator
        self._sigma_approximator = sigma_approximator

        self._delta_a = to_float_tensor(.5 * (max_a - min_a)).to(self.device)
        self._central_a = to_float_tensor(.5 * (max_a + min_a)).to(self.device)

        self._log_std_min = to_parameter(log_std_min)
        self._log_std_max = to_parameter(log_std_max)

        self._numeric_eps = 1e-6

        # Dynamics system of the agent
        self._control_system = control_system

        # Action space used for rescaling the action after atacom transformation
        # self._action_low = torch.from_numpy(mdp_info.action_space.low)
        # self._action_high = torch.from_numpy(mdp_info.action_space.high)

        # Assumes that reasonable cost is roughly capped at 1
        # self._cbf_scale = 1 / (1 - mdp_info.gamma)

        # Neural Networks for the CBF
        self._constraint_value_function_approximator = constraint_value_function_approximator
        self._constraint_approximator = constraint_approximator

        # ATACOM gain for error correction
        self._atacom_lam = to_parameter(atacom_lam)
        # ATACOM shape of slack function (how close to the constraint the agent is allowed)
        self._atacom_beta = to_parameter(atacom_beta)

        # Cvar margin
        self._margin = norm.pdf(norm.ppf(1 - accepted_risk)) / accepted_risk

        self._delta = delta
        self._delta_value_function = delta_value_function

        self._target_entropy = torch.tensor(target_entropy)

        self._debug_cbf_bound = []
        self._debug_residual_median = []
        self._debug_residual_max = []
        self._debug_auxiliary_action = []
        self._debug_constraint_violations = []
        self._debug_J_k_variance = []
        self._debug_J_k_norm = []

        self._learn_constraint = False
        self._learn_cbf = True

        self._analytical_const = analytical_const
        self.K = 0.5
        self.derivation_step_size = 1e-4

        self._atacom_dc = atacom_dc
        self._use_viability = use_viability

        self._add_save_attr(
            _mu_approximator='mushroom',
            _sigma_approximator='mushroom',
            _delta_a='torch',
            _central_a='torch',
            _log_std_min='mushroom',
            _log_std_max='mushroom',
            _numeric_eps='primitive',
            _control_system='mushroom',
            # _cbf_scale='primitive',
            _atacom_lam='mushroom',
            _atacom_beta='mushroom',
            _margin='primitive',
        )

    def draw_action(self, state):
        action = self.compute_action_and_log_prob_t(np.atleast_2d(state), return_log_prob=False)
        return action.detach().cpu().numpy()

    def apply_atacom(self, alpha, state):
        alpha_clipped = torch.clamp(alpha, -1, 1)

        B_u, b = self._get_atacom_transformations(alpha, state)

        tangential_term = B_u @ alpha_clipped[:, :, None]

        u = tangential_term + b

        action = u[:, :alpha.shape[1]].squeeze(0, 2)

        action_scale = torch.minimum(torch.min(1 / torch.abs(action), axis=-1)[0],
                                     torch.tensor([1], device=self.device)).detach()
        if action.ndim > 1:
            action_scale = action_scale.unsqueeze(1)
        scaled_action = action * action_scale

        return scaled_action, B_u

    def J_slack(self, slack):
        out = np.zeros(slack.shape + (slack.shape[1],))
        np.einsum('ijj->ij', out)[:] = 1 / np.maximum(np.exp(-self._atacom_beta() * slack), 1e-5) - 1

        return out

    def _get_atacom_transformations(self, alpha, state):
        q = self._control_system.get_q(state)  # (B, q)
        x = self._control_system.get_x(state)
        x_dot = self._control_system.get_x_dot(state)
        cons, J_k, J_x = self.compute_constraint_and_grad(q, x)  # (B, k), (B, k, q), (B, k, x)

        f = self._control_system.f(q)

        # Check if G needs alpha
        if len(signature(self._control_system.G).parameters) == 1:
            G = self._control_system.G(q)
        else:
            G = self._control_system.G(q, alpha)

        drift = np.zeros(cons.shape + (1,))

        if self._analytical_const:
            anal_cons, anal_J_k, anal_J_x = self._analytical_const(q, x)

            # Need to apply viability constraints to the analytical part
            state_dim = anal_J_k.shape[2] // 2

            # Move q jacobian to q_dot because we assume second order dynamics for viability constraints
            anal_J_k_q = anal_J_k[:, :, :state_dim]

            anal_J_k_new = np.zeros_like(anal_J_k)
            anal_J_k_new[:, :, state_dim:] = anal_J_k_q

            q_dot = q[:, state_dim:, None]

            # Transform to viability constraint
            anal_cons = anal_cons + self.K * (anal_J_k_q @ q_dot).squeeze(2)

            cons = np.concatenate((anal_cons, cons), axis=1)
            J_k = np.concatenate((anal_J_k_new, J_k), axis=1)
            J_x = np.concatenate((anal_J_x, J_x), axis=1)

            q_delta = q.copy()
            q_delta[:, :state_dim] += q_delta[:, state_dim:] * self.derivation_step_size
            _, anal_J_k_delta, _ = self._analytical_const(q, x)

            anal_J_k_q_dot = (anal_J_k_q - anal_J_k_delta[:, :, :state_dim]) / self.derivation_step_size

            drift[:, :anal_J_k.shape[1]] = (anal_J_k_q + anal_J_k_q_dot) @ q_dot
            
        elif self._use_viability and self._constraint_approximator is not None:
            # Assumes states is stacked as [q, q_dot]
            state_dim = q.shape[-1] // 2
            q_dot = q[:, state_dim:, None]

            J_k_viability = J_k[..., :state_dim]

            # Transform to viability constraint
            cons = cons + self.K * (J_k_viability @ q_dot).squeeze(2)

            # q_delta = q.copy()
            # q_delta[:, :state_dim] += q_delta[:, state_dim:] * self.derivation_step_size
            # _, J_k_delta, _ = self.compute_constraint_and_grad(q_delta, x)
            # J_k_viability_delta = J_k_delta[..., :state_dim]
            # J_k_dot = (J_k_viability_delta - J_k_viability) / self.derivation_step_size

            # drift += (J_k_viability + J_k_dot) @ q_dot

        lam = self._atacom_lam()

        slack = np.maximum(-cons, 1e-5)

        J_G = J_k @ G  # (B, k, u)
    
        drift += J_k @ f  # (B, k)

        drift = np.maximum(drift, 0)

        J_u = np.concatenate((J_G, self.J_slack(slack)), axis=-1)

        useful_constr = self._directional_constraints(drift.squeeze(-1), J_G, alpha)
        dir_slack = slack.copy()
        dir_slack[np.logical_not(useful_constr)] = 1e+2

        J_u_dir = np.concatenate((J_G, self.J_slack(dir_slack)), axis=-1)  # (B, k, u + k)

        B_u = batch_smooth_basis(J_u_dir)[..., :alpha.shape[-1]]

        c = cons + slack

        # J_u_inv = torch.linalg.pinv(J_u, atol=0.1)
        J_u_inv = np.linalg.pinv(J_u)

        if not (0 in J_x.shape and 0 in x_dot.shape):
            uncontrollable = J_u_inv @ J_x @ x_dot
        else:
            uncontrollable = 0
        drift_compensation = J_u_inv @ drift
        contraction_term = lam * J_u_inv @ c[:, :, None]

        b = -uncontrollable - drift_compensation - contraction_term

        # CAST ABACK TO TORCH
        return torch.from_numpy(B_u).to(self.device), torch.from_numpy(b).to(self.device)

    def compute_constraint_and_grad(self, q, x):
        q_tensor = torch.as_tensor(q, dtype=torch.float32, device=self.device)
        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device)

        state = torch.hstack([q_tensor, x_tensor])

        def dummy_aux_fun(x):
            out = self.compute_constraint(x)
            return out, out.detach()

        def get_vjp(el):
            return torch.func.jacrev(dummy_aux_fun, argnums=0, has_aux=True)(el)

        J, cons = torch.vmap(get_vjp)(state)

        J_q = J[:, :, :self._control_system.dim_q].detach()
        J_x = J[:, :, self._control_system.dim_q:].detach()

        return cons.double().cpu().numpy(), J_q.double().cpu().numpy(), J_x.double().cpu().numpy()

    def compute_constraint(self, state):

        if self._constraint_approximator is None:
            cbf = torch.tensor([]).to(state.device)
        else:
            cbf = self.constraint_value(state) - self._delta().detach()

        if self._constraint_value_function_approximator is None:
           cbf_value_function = torch.tensor([]).to(state.device)
        else:
            cbf_value_function = self.constraint_value_function(state) - self._delta_value_function().detach()

        return torch.concatenate((cbf, cbf_value_function))

    def constraint_value(self, state):
        cbf_mean, cbf_log_std = self._constraint_approximator.model.network(state)
        cbf_log_std = torch.clamp(cbf_log_std, self._log_std_min(), self._log_std_max())
        return cbf_mean + cbf_log_std.exp() * self._margin

    def constraint_value_function(self, state):
        cbf_mean, cbf_log_std = self._constraint_value_function_approximator.model.network(state)
        cbf_log_std = torch.clamp(cbf_log_std, self._log_std_min(), self._log_std_max())
        return cbf_mean + cbf_log_std.exp() * self._margin

    def compute_action_and_log_prob(self, state):
        """
        Function that samples actions using the reparametrization trick and the log probability for such actions.

        Args:
            state (np.ndarray): the state in which the action is sampled.

        Returns:
            The actions sampled and the log probability as numpy arrays.

        """
        a, log_prob = self.compute_action_and_log_prob_t(state)
        return a.detach().cpu().numpy(), log_prob.detach().cpu().numpy()

    def compute_action_and_log_prob_t(self, state, return_log_prob=True):
        """
        Function that samples actions using the reparametrization trick and, optionally, the log probability for such
        actions.

        Args:
            state (np.ndarray): the state in which the action is sampled;
            compute_log_prob (bool, True): whether to compute the log  probability or not.

        Returns:
            The actions sampled and, optionally, the log probability as torch tensors.

        """
        dist = self.distribution(state)
        a_raw = dist.rsample()
        a = torch.tanh(a_raw)
        alpha = a * self._delta_a + self._central_a

        alpha = alpha.double()

        action, B = self.apply_atacom(alpha, state)

        if return_log_prob:
            log_prob = dist.log_prob(a_raw).sum(dim=1)
            log_prob -= torch.log(1. - a.pow(2) + self._numeric_eps).sum(dim=1)

            _, N, M = B.size()
            K = min(N, M)

            # Add delta for numerical stability
            delta = torch.eye(K) * 1e-4
            log_det_B = torch.logdet(B[:, :K, :K] + delta.to(B))

            log_prob -= torch.maximum(log_det_B, self._target_entropy)

            return action, log_prob
        return action

    def distribution(self, state):
        """
        Compute the policy distribution in the given states.

        Args:
            state (np.ndarray): the set of states where the distribution is computed.

        Returns:
            The torch distribution for the provided states.

        """
        mu = self._mu_approximator.predict(state, output_tensor=True)
        log_sigma = self._sigma_approximator.predict(state, output_tensor=True)
        # Bound the log_std
        log_sigma = torch.clamp(log_sigma, self._log_std_min(), self._log_std_max())
        return torch.distributions.Normal(mu, log_sigma.exp())

    def entropy(self, state=None):
        """
        Compute the entropy of the policy.

        Args:
            state (np.ndarray): the set of states to consider.

        Returns:
            The value of the entropy of the policy.

        """
        _, log_pi = self.compute_action_and_log_prob(state)
        return -log_pi.mean()

    def set_weights(self, weights):
        """
        Setter.

        Args:
            weights (np.ndarray): the vector of the new weights to be used by the policy.

        """
        mu_weights = weights[:self._mu_approximator.weights_size]
        sigma_weights = weights[self._mu_approximator.weights_size:]

        self._mu_approximator.set_weights(mu_weights)
        self._sigma_approximator.set_weights(sigma_weights)

    def get_weights(self):
        """
        Getter.

        Returns:
             The current policy weights.

        """
        mu_weights = self._mu_approximator.get_weights()
        sigma_weights = self._sigma_approximator.get_weights()

        return np.concatenate([mu_weights, sigma_weights])

    @property
    def use_cuda(self):
        """
        True if the policy is using cuda_tensors.
        """
        return self._mu_approximator.model.use_cuda

    def parameters(self):
        """
        Returns the trainable policy parameters, as expected by torch optimizers.

        Returns:
            List of parameters to be optimized.

        """
        return chain(self._mu_approximator.model.network.parameters(),
                     self._sigma_approximator.model.network.parameters())

    def _directional_constraints(self, drift, J_G, alpha):
        if self._atacom_dc:
            constraint_direction = (J_G @ alpha.detach().numpy()[..., None]).squeeze(-1)
            useful_constr = constraint_direction > 0
        else:
            useful_constr = np.ones(drift.shape, dtype=bool)
        return useful_constr


class DatacomSAC(DeepAC):
    """
    Soft Actor-Critic algorithm.
    "Soft Actor-Critic Algorithms and Applications".
    Haarnoja T. et al.. 2019.

    """

    def __init__(self, mdp_info, control_system, accepted_risk, actor_mu_params, actor_sigma_params, actor_optimizer,
                 critic_params, batch_size, initial_replay_size, max_replay_size, warmup_transitions, tau, lr_alpha,
                 cost_budget, constraint_params, constr_aggregation, constr_aggregation_value_function, atacom_lam, atacom_beta, lr_delta, init_delta, atacom_dc, use_viability, n_learnable_constr,
                 delta_warmup_transitions, violation_memory_ratio, analytical_constraint=None, constraint_weight_decay=None,
                 use_log_alpha_loss=False, log_std_min=-20, log_std_max=2, target_entropy=None, critic_fit_params=None):
        """
        Constructor.

        Args:
            actor_mu_params (dict): parameters of the actor mean approximator to build;
            actor_sigma_params (dict): parameters of the actor sigma approximator to build;
            actor_optimizer (dict): parameters to specify the actor optimizer algorithm;
            critic_params (dict): parameters of the critic approximator to build;
            batch_size ((int, Parameter)): the number of samples in a batch;
            initial_replay_size (int): the number of samples to collect before starting the learning;
            max_replay_size (int): the maximum number of samples in the replay memory;
            warmup_transitions ([int, Parameter]): number of samples to accumulate in the replay memory to start the
                policy fitting;
            tau ([float, Parameter]): value of coefficient for soft updates;
            lr_alpha ([float, Parameter]): Learning rate for the entropy coefficient;
            use_log_alpha_loss (bool, False): whether to use the original implementation loss or the one from the
                paper;
            log_std_min ([float, Parameter]): Min value for the policy log std;
            log_std_max ([float, Parameter]): Max value for the policy log std;
            target_entropy (float, None): target entropy for the policy, if None a default value is computed;
            critic_fit_params (dict, None): parameters of the fitting algorithm of the critic approximator.

        """
        self._critic_fit_params = dict() if critic_fit_params is None else critic_fit_params

        self._batch_size = to_parameter(batch_size)
        self._warmup_transitions = to_parameter(warmup_transitions)
        self._tau = to_parameter(tau)

        self._use_log_alpha_loss = use_log_alpha_loss

        # Maximum acceptable sum of cost per episode
        self._cost_budget = cost_budget / mdp_info.horizon * \
                            (1 - mdp_info.gamma ** mdp_info.horizon) / (1 - mdp_info.gamma)

        # Sum of cost of current episode
        self._episode_step_count = 0
        self._episode_costs = []
        self._episode_states = []

        self._episode_end = False

        if target_entropy is None:
            self._target_entropy = -np.prod(mdp_info.action_space.shape).astype(np.float32)
        else:
            self._target_entropy = target_entropy

        self._replay_memory = SafeReplayMemory(initial_replay_size, max_replay_size)

        self._violation_memory_ratio = violation_memory_ratio
        self._violation_replay_memory = ReplayMemory(0, int(max_replay_size * self._violation_memory_ratio))

        if 'n_models' in critic_params.keys():
            assert critic_params['n_models'] == 2
        else:
            critic_params['n_models'] = 2

        constraint_params['n_fit_targets'] = 5

        constraint_value_function_params = deepcopy(constraint_params)

        target_critic_params = deepcopy(critic_params)
        self._critic_approximator = Regressor(TorchApproximator, **critic_params)
        self._target_critic_approximator = Regressor(TorchApproximator, **target_critic_params)

        self._init_target(self._critic_approximator, self._target_critic_approximator)

        self.constr_aggregation = constr_aggregation
        self.constr_aggregation_value_function = constr_aggregation_value_function

        if self.constr_aggregation and self.constr_aggregation != 'none':
            constraint_params["loss"] = self.gaussian_wasserstein_dist

            if self.constr_aggregation == 'array':
                constr_shape = n_learnable_constr
            else:
                constr_shape = 1
            constraint_params["output_shape"] = (constr_shape, 2)
            if constraint_weight_decay and constraint_weight_decay > 0:
                constraint_params["optimizer"]["params"]["weight_decay"] = constraint_weight_decay
            target_constraint_params = deepcopy(constraint_params)
            self._constraint_approximator = Regressor(TorchApproximator, **constraint_params)
            self._target_constraint_approximator = Regressor(TorchApproximator, **target_constraint_params)
            self._init_target(self._constraint_approximator, self._target_constraint_approximator)
        else:
            self._constraint_approximator = None
            self._target_constraint_approximator = None

        if self.constr_aggregation_value_function and self.constr_aggregation_value_function != 'none':
            constraint_value_function_params["loss"] = self.gaussian_wasserstein_dist_value_function

            if self.constr_aggregation_value_function == 'array':
                constr_shape = n_learnable_constr
            else:
                constr_shape = 1
            constraint_value_function_params["output_shape"] = (constr_shape, 2)
            constraint_value_function_params["output_mod"] = torch.nn.Softplus()
            target_constraint_value_function_params = deepcopy(constraint_value_function_params)

            self._constraint_value_function_approximator = Regressor(TorchApproximator, **constraint_value_function_params)
            self._target_constraint_value_function_approximator = Regressor(TorchApproximator, **target_constraint_value_function_params)
            self._init_target(self._constraint_value_function_approximator, self._target_constraint_value_function_approximator)
        else:
            self._constraint_value_function_approximator = None
            self._target_constraint_value_function_approximator = None

        # self.lr_schedueler = optim.lr_scheduler.LambdaLR(self._constraint_approximator.model._optimizer,
        #                                                  lambda step: 0.97 ** (step // 10000))
        actor_mu_approximator = Regressor(TorchApproximator, **actor_mu_params)
        actor_sigma_approximator = Regressor(TorchApproximator, **actor_sigma_params)

        policy = DatacomSACPolicy(actor_mu_approximator, actor_sigma_approximator, self._constraint_value_function_approximator,
                                  self._constraint_approximator, control_system, mdp_info, accepted_risk, self.delta, self.delta_value_function,
                                  atacom_lam,
                                  atacom_beta, self._target_entropy, mdp_info.action_space.low,
                                  mdp_info.action_space.high,
                                  log_std_min, log_std_max, atacom_dc, use_viability, analytical_constraint)

        self._log_alpha = torch.tensor(1., dtype=torch.float32).log()

        def softplus_inverse(y):
            y = torch.tensor(y, dtype=torch.float32)
            return y + torch.log1p(-torch.exp(-y))

        if delta_warmup_transitions > 0 and lr_delta < 0:
            self._delta_value = softplus_inverse(3.)
            self._delta_value_function = softplus_inverse(3.)
            self._delta_init = softplus_inverse(init_delta)
        else:
            self._delta_value = softplus_inverse(init_delta)
            self._delta_value_function = softplus_inverse(init_delta)


        self._log_alpha.requires_grad_()
        self._delta_value.requires_grad_()
        self._delta_value_function.requires_grad_()

        if lr_alpha >= 0:
            self._alpha_optim = optim.Adam([self._log_alpha], lr=lr_alpha)

        if lr_delta >= 0:
            self._delta_optim = optim.Adam([self._delta_value], lr=lr_delta)
            self._delta_optim_value_function = optim.Adam([self._delta_value_function], lr=lr_delta)
        
        self._delta_warmup_transitions = delta_warmup_transitions

        policy_parameters = chain(actor_mu_approximator.model.network.parameters(),
                                  actor_sigma_approximator.model.network.parameters())

        self.training_loss = []
        self.training_loss_value_function = []
        self.cbf_reg_loss = []

        self._loss_critic = []

        self._add_save_attr(
            _critic_fit_params='pickle',
            _batch_size='mushroom',
            _warmup_transitions='mushroom',
            _tau='mushroom',
            _cost_budget='primitive',
            _target_entropy='primitive',
            _replay_memory='mushroom',
            _violation_replay_memory='mushroom',
            _critic_approximator='mushroom',
            _target_critic_approximator='mushroom',
            _constraint_approximator='mushroom',
            _target_constraint_approximator='mushroom',
            _constraint_value_function_approximator='mushroom',
            _target_constraint_value_function_approximator='mushroom',
            _use_log_alpha_loss='primitive',
            _log_alpha='torch',
            _delta_value='torch',
            _alpha_optim='torch',
            _delta_optim='torch',
            _delta_warmup_transitions='primitive',
        )

        super().__init__(mdp_info, policy, actor_optimizer, policy_parameters)

    def fit(self, dataset, **info):
        self._add_episode_cost(dataset)

        if hasattr(self, "_delta_init") and self._replay_memory.size == self._delta_warmup_transitions:
            self._delta_value = self._delta_init.clone()
            self._delta_value_function = self._delta_init.clone()

        if (hasattr(self, "_delta_optim_value_function") or hasattr(self, "_delta_optim")) and dataset[-1][-1] and self._replay_memory.size > self._delta_warmup_transitions:
            self.update_delta()

        self._replay_memory.add(dataset)

        self._violation_replay_memory.add(self._get_violations(dataset))

        if self._replay_memory.initialized:
            state, action, reward, next_state, cost, absorbing, _ = self._replay_memory.get(self._batch_size())

            if self._replay_memory.size > self._warmup_transitions():
                action_new, log_prob_new = self.policy.compute_action_and_log_prob_t(state)
                loss = self._loss(state, action_new, log_prob_new)
                self._optimize_actor_parameters(loss)
                if hasattr(self, "_alpha_optim"):
                    self._update_alpha(log_prob_new.detach())

            q_next = self._next_q(next_state, absorbing)
            q = reward + self.mdp_info.gamma * q_next

            self._critic_approximator.fit(state, action, q, **self._critic_fit_params)
            self._loss_critic.append(np.mean([self._critic_approximator.model[i].loss_fit.copy() for i in range(len(self._critic_approximator.model))]))

            # Fit CBF
            if self._violation_replay_memory.size > int(self._batch_size() * self._violation_memory_ratio):
                violating_state, _, violating_cost, violating_next_state, violating_absorbing, _ = self._violation_replay_memory.get(
                    int(self._batch_size() * self._violation_memory_ratio))

                state = np.concatenate([state, violating_state], axis=0)
                cost = np.concatenate([cost, violating_cost], axis=0)
                next_state = np.concatenate([next_state, violating_next_state], axis=0)
                absorbing = np.concatenate([absorbing, violating_absorbing], axis=0)

            constraint_state, next_constraint_state = self.to_constraint_state(state, next_state)

            error_value_function = torch.tensor(np.maximum(constr_aggregation_mapping[self.constr_aggregation_value_function](cost, axis=1), 0), dtype=torch.float32, device=self.device).flatten()
            error = torch.tensor(constr_aggregation_mapping[self.constr_aggregation](cost, axis=1), dtype=torch.float32, device=self.device).flatten()

            def fit_cbf(self, approximator, target_approximator, loss_function, training_loss,
                            error, constraint_state, next_constraint_state, absorbing):
                with torch.no_grad():
                    next_mu, next_log_std = target_approximator.predict(
                        next_constraint_state, output_tensor=True)

                state_tensor = torch.tensor(constraint_state, dtype=torch.float32, device=self.device)

                approximator.model._optimizer.zero_grad()

                pred = approximator.predict(state_tensor, output_tensor=True)

                absorbing = np.tile(absorbing, error.shape[0] // absorbing.shape[0])

                loss = loss_function(pred, error, next_mu, next_log_std,
                                        torch.tensor(absorbing, dtype=torch.float32, device=self.device),
                                        torch.ones_like(error, device=self.device) * self.mdp_info.gamma)

                loss.backward()

                approximator.model._optimizer.step()

                training_loss.append(loss.detach().item())

                self._update_target(approximator, target_approximator)

            if self._constraint_approximator is not None and self._target_constraint_approximator is not None:
                fit_cbf(self, self._constraint_approximator, self._target_constraint_approximator, self.gaussian_wasserstein_dist, self.training_loss,
                            error, constraint_state, next_constraint_state, absorbing)

            if self._constraint_value_function_approximator is not None and self._target_constraint_value_function_approximator is not None:
                fit_cbf(self, self._constraint_value_function_approximator, self._target_constraint_value_function_approximator, self.gaussian_wasserstein_dist_value_function, self.training_loss_value_function,\
                            error_value_function, constraint_state, next_constraint_state, absorbing)

            self._update_target(self._critic_approximator, self._target_critic_approximator)

    def _add_episode_cost(self, dataset):
        for sample in dataset:
            if self._episode_step_count == 0:
                self._episode_costs = [sample[4]]
                self._episode_states = [sample[0]]

            self._episode_costs.append(sample[4])
            self._episode_states.append(sample[3])
            self._episode_step_count += 1

            if sample[-1]:
                self._episode_step_count = 0

    def to_constraint_state(self, state, next_state):
        q = self.policy._control_system.get_q(state)
        x = self.policy._control_system.get_x(state)
        constraint_state = np.hstack([q, x])

        q_next = self.policy._control_system.get_q(next_state)
        x_next = self.policy._control_system.get_x(next_state)
        next_constraint_state = np.hstack([q_next, x_next])

        return constraint_state, next_constraint_state

    def _get_violations(self, dataset):
        new_data = []
        for el in dataset:
            if (el[4] > 0).any():
                new_data.append(el[:2] + (el[4], el[3]) + el[5:])

        return new_data

    def _loss(self, state, action_new, log_prob):
        q_0 = self._critic_approximator(state, action_new, output_tensor=True, idx=0)
        q_1 = self._critic_approximator(state, action_new, output_tensor=True, idx=1)

        q = torch.min(q_0, q_1)

        return (self._alpha * log_prob - q).mean()

    def _update_alpha(self, log_prob):
        if self._use_log_alpha_loss:
            alpha_loss = - (self._log_alpha * (log_prob + self._target_entropy)).mean()
        else:
            alpha_loss = - (self._alpha * (log_prob + self._target_entropy)).mean()
        self._alpha_optim.zero_grad()
        alpha_loss.backward()
        self._alpha_optim.step()

    def episode_start(self):
        super().episode_start()
        self._episode_end = True

    def update_delta(self):
        epi_states, _ = self.to_constraint_state(self._episode_states, self._episode_states)

        if self._constraint_approximator is not None:
            cost_epi = self.policy.constraint_value(torch.tensor(epi_states).to(self.device)).detach().cpu() - self.delta()
            self._delta_optim.zero_grad()

            costs = torch.tensor(constr_aggregation_mapping[self.constr_aggregation](np.array(self._episode_costs)).copy(), dtype=torch.float) - self._cost_budget
            loss = F.smooth_l1_loss(costs.flatten(), cost_epi.flatten(), reduction='mean')

            loss.backward()
            self._delta_optim.step()
        
        if self._constraint_value_function_approximator is not None:
            cost_value_function_epi = self.policy.constraint_value_function(torch.tensor(epi_states).to(self.device)).detach().cpu() - self.delta_value_function()
            self._delta_optim_value_function.zero_grad()

            cum_cost = 0
            cum_costs = []
            for cost in np.array(self._episode_costs)[::-1]:
                cum_cost = np.maximum(constr_aggregation_mapping[self.constr_aggregation_value_function](cost), 0) + self.mdp_info.gamma * cum_cost
                cum_costs.append(cum_cost)
            cum_costs = np.array(cum_costs)[::-1]

            costs_value_function = torch.tensor(cum_costs.copy(), dtype=torch.float) - self._cost_budget
            loss_value_function = F.smooth_l1_loss(costs_value_function.flatten(), cost_value_function_epi.flatten(), reduction='mean') 

            loss_value_function.backward()
            self._delta_optim_value_function.step()
            

    def _next_q(self, next_state, absorbing):
        """
        Args:
            next_state (np.ndarray): the states where next action has to be evaluated;
            absorbing (np.ndarray): the absorbing flag for the states in ``next_state``.

        Returns:
            Action-values returned by the critic for ``next_state`` and the action returned by the actor.

        """
        a, log_prob_next = self.policy.compute_action_and_log_prob(next_state)

        q = self._target_critic_approximator.predict(
            next_state, a, prediction='min') - self._alpha_np * log_prob_next
        q *= 1 - absorbing

        return q

    def _post_load(self):
        self._update_optimizer_parameters(self.policy.parameters())
        self.policy._constraint_approximator = self._constraint_approximator
        self.policy._constraint_value_function_approximator = self._constraint_value_function_approximator
        self.policy._delta = self.delta
        self.policy._target_entropy = torch.tensor(self._target_entropy)

        self.policy._debug_cbf_bound = []
        self.policy._debug_residual_median = []
        self.policy._debug_residual_max = []
        self.policy._debug_auxiliary_action = []
        self.policy._debug_constraint_violations = []
        self.policy._debug_J_k_variance = []
        self.policy._debug_J_k_norm = []
        self.policy.device = torch.device('cpu')
        self.policy._analytical_const = None

        self.training_loss = []
        self.training_loss_value_function = []

        self._episode_cost = 0
        self._episode_end = False

    # No property because it needs to be passed to policy
    def delta(self):
        return F.softplus(self._delta_value)

    def delta_value_function(self):
        return F.softplus(self._delta_value_function)

    @property
    def device(self):
        return self.policy.device

    @property
    def _alpha(self):
        return self._log_alpha.exp()

    @property
    def _alpha_np(self):
        return self._alpha.detach().cpu().numpy()

    @staticmethod
    def gaussian_wasserstein_dist_value_function(predicted_cost, cost, next_mean, next_log_std, absorbing, gamma, reduction='mean'):
        next_var = (2 * next_log_std).exp()

        predicted_mean = predicted_cost[0]
        predicted_mean_detach = predicted_mean.detach()

        target_mean = cost + (1 - absorbing) * gamma * next_mean
        J_mu = F.mse_loss(predicted_mean, target_mean, reduction=reduction)

        target_var_not_absorbing = cost ** 2 \
                                   + 2 * cost * gamma * next_mean \
                                   + gamma ** 2 * (next_var + next_mean ** 2) \
                                   - predicted_mean_detach ** 2

        target_var_not_absorbing = torch.clamp(target_var_not_absorbing, 1e-4, 1e8)

        target_var_absorbing = (cost - predicted_mean_detach) ** 2

        target_var = (1 - absorbing) * target_var_not_absorbing + absorbing * target_var_absorbing

        J_sigma = F.mse_loss(predicted_cost[1].exp(), torch.sqrt(target_var), reduction=reduction)

        return J_mu + J_sigma

    @staticmethod
    def gaussian_wasserstein_dist(predicted_cost, cost, next_mean, next_log_std, absorbing, gamma, reduction='mean'):
        next_var = (2 * next_log_std).exp()

        predicted_mean = predicted_cost[0]
        predicted_mean_detach = predicted_mean.detach()

        target_mean = cost
        
        J_mu = F.mse_loss(predicted_mean, target_mean, reduction=reduction)

        target_var = (cost - predicted_mean_detach) ** 2

        J_sigma = F.mse_loss(predicted_cost[1].exp(), torch.sqrt(target_var), reduction=reduction)

        return J_mu + J_sigma