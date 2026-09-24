import numpy as np
import torch
import torch.optim as optim
from mushroom_rl.utils.preprocessors import MinMaxPreprocessor
from mushroom_rl.algorithms.actor_critic.deep_actor_critic import TD3
from mushroom_rl.policy import ClippedGaussianPolicy

from directional_atacom.algorithms import *
from directional_atacom.utils.networks import *

from directional_atacom.utils.beta_policy import BetaPolicy


def agent_builder(alg, mdp, control_system, normalize_state, **kwargs):
    for key in ["n_features_actor", "n_features_critic", "n_features_constraint"]:
        if key in kwargs.keys():
            if isinstance(kwargs[key], list):
                kwargs[key] = ' '.join(map(str, kwargs[key]))
    
    alg = alg.replace("_dc", "")
    alg = alg.replace("_vel", "") 

    if alg == "td3":
        agent = build_td3(mdp, **kwargs)

    if alg == "sac":
        agent = build_sac(mdp, **kwargs)

    if alg == "datacom_sac":
        agent = build_datacom_sac(mdp, control_system, **kwargs)

    if alg == "cbf_sac":
        agent = build_cbf_sac(mdp, control_system, **kwargs)

    if alg == "iqn_datacom_sac":
        agent = build_iqn_datacom_sac(mdp, control_system, **kwargs)

    if alg == "atacom_sac":
        agent = build_atacom_sac(mdp, control_system, **kwargs)

    if alg == "safelayer_td3":
        agent = build_safelayer_td3(mdp, **kwargs)

    if alg == "lag_sac":
        agent = build_lag_sac(mdp, **kwargs)

    if alg == "wc_lag_sac":
        agent = build_wcsac(mdp, **kwargs)

    if normalize_state:
        if hasattr(agent, "add_state_preprocessor"):
            agent.add_state_preprocessor(MinMaxPreprocessor(mdp.info))
        else:
            agent.add_preprocessor(MinMaxPreprocessor(mdp.info))

    return agent


def apply_beta_policy(agent, mdp, actor_mu_params, actor_sigma_params, actor_optimizer):
    """Replace the agent's Gaussian policy with a Beta policy over the action box."""
    from mushroom_rl.approximators import Regressor
    from mushroom_rl.approximators.parametric import TorchApproximator
    from itertools import chain

    policy_params = {
        'min_a': mdp.info.action_space.low,
        'max_a': mdp.info.action_space.high
    }

    actor_alpha_approximator = Regressor(TorchApproximator, **actor_mu_params)
    actor_beta_approximator = Regressor(TorchApproximator, **actor_sigma_params)

    policy_parameters = chain(actor_alpha_approximator.model.network.parameters(),
                              actor_beta_approximator.model.network.parameters())
    agent.policy = BetaPolicy(alpha_approximator=actor_alpha_approximator,
                              beta_approximator=actor_beta_approximator,
                              **policy_params)
    agent._optimizer = actor_optimizer['class'](policy_parameters, **actor_optimizer['params'])

    return agent


def build_atacom_sac(mdp, control_system, slack_limit, atacom_lam, atacom_beta, atacom_dc, initial_replay_size, max_replay_size,
                              batch_size, n_features_actor, n_features_critic, activation,
                              learning_rate_actor, learning_rate_critic, tau, lr_alpha, init_alpha, target_entropy, 
                              warmup_transitions, use_viability, use_cuda, beta_policy,
                              **kwargs):
    actor_mu_params, actor_sigma_params, actor_optimizer, critic_params, alg_params = \
        build_sac_params(mdp, n_features_actor, n_features_critic, learning_rate_actor, learning_rate_critic, activation,
                         use_cuda, tau, lr_alpha, target_entropy, warmup_transitions)

    constraint_func = mdp.constraint_func

    action_filter_ratio = getattr(mdp, "action_filter_ratio", None)
    save_prev_action = mdp.save_prev_action_obs if hasattr(mdp, "save_prev_action_obs") else None

    if mdp.__class__.__name__.startswith("AirHockey"):
        atacom_beta = [atacom_beta] * mdp.original_constraint_list.output_dim()
        atacom_beta[-4] = atacom_beta[-3] = 0.01

    agent = AtacomSAC(mdp.info, control_system, slack_limit, atacom_lam, atacom_beta, atacom_dc, constraint_func, use_viability,
                              actor_mu_params,
                              actor_sigma_params,
                              actor_optimizer, critic_params,
                              **alg_params,
                              initial_replay_size=initial_replay_size, max_replay_size=max_replay_size,
                              batch_size=batch_size, init_alpha=init_alpha, action_filter_ratio=action_filter_ratio, save_prev_action=save_prev_action)

    if beta_policy:
        apply_beta_policy(agent, mdp, actor_mu_params, actor_sigma_params, actor_optimizer)

    return agent


def build_constraint(control_system, constraint_distribution, learning_rate_constraint,
                     n_features_constraint, use_cuda):
    if constraint_distribution == "None":
        constraint_params = dict(network=MLP,
                                 optimizer={'class': optim.Adam,
                                            'params': {'lr': learning_rate_constraint}},
                                 n_features=list(map(int, n_features_constraint.split(' '))),
                                 input_shape=(control_system.dim_q + control_system.dim_x,),
                                 output_shape=(1, 2),
                                 use_cuda=use_cuda,
                                 quiet=True,
                                 loss=F.mse_loss,
                                 activation='relu')

    elif constraint_distribution == "gaussian":
        constraint_params = dict(network=GaussianConstraintNetwork,
                                 optimizer={'class': optim.Adam,
                                            'params': {'lr': learning_rate_constraint}},
                                 n_features=list(map(int, n_features_constraint.split(' '))),
                                 input_shape=(control_system.dim_q + control_system.dim_x,),
                                 output_shape=(1, 2),
                                 use_cuda=use_cuda,
                                 n_fit_targets=5,
                                 quiet=True,
                                 activation='sigmoid')

    elif constraint_distribution == "quantile":
        constraint_params = dict(network=QuantileCriticNetwork,
                                 optimizer={'class': optim.Adam,
                                            'params': {'lr': learning_rate_constraint}},
                                 n_features=list(map(int, n_features_constraint.split(' '))),
                                 use_cuda=use_cuda,
                                 input_shape=(control_system.dim_q + control_system.dim_x,),
                                 output_shape=(1,),
                                 n_fit_targets=1,
                                 quiet=True,
                                 activation='relu')

    return constraint_params


def build_td3_params(mdp, n_features_actor, n_features_critic, learning_rate_actor, learning_rate_critic, use_cuda):
    # policy_class = OrnsteinUhlenbeckPolicy
    # policy_params = dict(sigma=np.ones(mdp.info.action_space.shape) * 0.2, theta=.15, dt=mdp.dt * mdp._n_sub_steps)

    policy_class = ClippedGaussianPolicy
    policy_params = dict(
        sigma=np.eye(mdp.info.action_space.shape[0]) * 0.25,
        low=mdp.info.action_space.low,
        high=mdp.info.action_space.high)

    # Settings
    tau = .001

    # Approximator
    actor_input_shape = mdp.info.observation_space.shape
    actor_params = dict(network=TD3ActorNetwork,
                        n_features=list(map(int, n_features_actor.split(' '))),
                        input_shape=actor_input_shape,
                        output_shape=mdp.info.action_space.shape,
                        action_scaling=(mdp.info.action_space.high - mdp.info.action_space.low) / 2,
                        use_cuda=use_cuda)

    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': learning_rate_actor}}

    critic_input_shape = (actor_input_shape[0] + mdp.info.action_space.shape[0],)
    critic_params = dict(network=TD3CriticNetwork,
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': learning_rate_critic}},
                         loss=F.mse_loss,
                         n_features=list(map(int, n_features_critic.split(' '))),
                         input_shape=critic_input_shape,
                         action_shape=mdp.info.action_space.shape,
                         output_shape=(1,),
                         action_scaling=(mdp.info.action_space.high - mdp.info.action_space.low) / 2,
                         use_cuda=use_cuda)

    return policy_class, policy_params, actor_params, actor_optimizer, critic_params, tau


def build_td3(mdp, initial_replay_size, max_replay_size, batch_size, n_features_actor, n_features_critic,
              learning_rate_actor, learning_rate_critic, **kwargs):
    use_cuda = False

    policy_class, policy_params, actor_params, actor_optimizer, critic_params, tau = build_td3_params(
        mdp, n_features_actor, n_features_critic, learning_rate_actor, learning_rate_critic, use_cuda)

    agent = TD3(mdp.info, policy_class, policy_params,
                actor_params, actor_optimizer, critic_params, batch_size,
                initial_replay_size, max_replay_size, tau)

    return agent


def build_safelayer_td3(mdp, initial_replay_size, max_replay_size, batch_size, n_features_actor, n_features_critic,
                        learning_rate_actor, learning_rate_critic, learning_rate_constraint, n_features_constraint,
                        delta, warmup_transitions, **kwargs):
    use_cuda = False

    constraint_params = dict(network=MLP,
                             optimizer={'class': optim.Adam,
                                        'params': {'lr': learning_rate_constraint}},
                             input_shape=mdp.info.observation_space.shape,
                             output_shape=mdp.info.action_space.shape,
                             n_features=list(map(int, n_features_constraint.split(' '))),
                             loss=SafeLayerTD3.safelayer_loss,
                             use_cuda=False,
                             quiet=True,
                             n_fit_targets=3)

    policy_class, policy_params, actor_params, actor_optimizer, critic_params, tau = build_td3_params(
        mdp, n_features_actor, n_features_critic, learning_rate_actor, learning_rate_critic, use_cuda)

    policy_class = ClippedGaussianPolicy

    agent = SafeLayerTD3(mdp.info, policy_class, policy_params, actor_params, actor_optimizer, critic_params,
                         batch_size,
                         initial_replay_size, max_replay_size, tau, delta, constraint_params, warmup_transitions)

    return agent


def build_sac_params(mdp, n_features_actor, n_features_critic, learning_rate_actor, learning_rate_critic, activation, use_cuda,
                     tau, lr_alpha, target_entropy, warmup_transitions):
    actor_mu_params = dict(network=SACActorNetwork,
                           input_shape=mdp.info.observation_space.shape,
                           output_shape=mdp.info.action_space.shape,
                           n_features=list(map(int, n_features_actor.split(' '))),
                           activation=activation,
                           use_cuda=use_cuda)
    actor_sigma_params = dict(network=SACActorNetwork,
                              input_shape=mdp.info.observation_space.shape,
                              output_shape=mdp.info.action_space.shape,
                              n_features=list(map(int, n_features_actor.split(' '))),
                              activation=activation,
                              use_cuda=use_cuda)

    actor_optimizer = {'class': optim.Adam,
                       'params': {'lr': learning_rate_actor}}

    critic_params = dict(network=SACCriticNetwork,
                         input_shape=(mdp.info.observation_space.shape[0] + mdp.info.action_space.shape[0],),
                         optimizer={'class': optim.Adam,
                                    'params': {'lr': learning_rate_critic}},
                         loss=F.mse_loss,
                         n_features=list(map(int, n_features_critic.split(' '))),
                         activation=activation,
                         output_shape=(1,),
                         action_shape=mdp.info.action_space.shape,
                         action_scaling=(mdp.info.action_space.high - mdp.info.action_space.low) / 2,
                         use_cuda=use_cuda
                         )

    alg_params = dict(warmup_transitions=warmup_transitions,
                      tau=tau,
                      lr_alpha=lr_alpha,
                      target_entropy=target_entropy)

    return actor_mu_params, actor_sigma_params, actor_optimizer, critic_params, alg_params

def build_sac(mdp, initial_replay_size, max_replay_size, batch_size, n_features_actor, n_features_critic,
              learning_rate_actor, learning_rate_critic, activation, use_cuda, tau, lr_alpha, target_entropy, warmup_transitions,
              init_alpha=None, beta_policy=False,
              **kwargs):
    actor_mu_params, actor_sigma_params, actor_optimizer, critic_params, alg_params = \
        build_sac_params(mdp, n_features_actor, n_features_critic, learning_rate_actor, learning_rate_critic, activation,
                         use_cuda, tau, lr_alpha, target_entropy, warmup_transitions)

    print(alg_params, use_cuda)

    agent = SAC(mdp.info, actor_mu_params, actor_sigma_params, actor_optimizer, critic_params, **alg_params,
                initial_replay_size=initial_replay_size, max_replay_size=max_replay_size,
                batch_size=batch_size,
                action_filter_ratio=getattr(mdp, "action_filter_ratio", None),
                save_prev_action=getattr(mdp, "save_prev_action_obs", None))

    if init_alpha is not None:
        agent._log_alpha = torch.tensor(np.log(init_alpha)).to(agent._log_alpha).requires_grad_(True)
        agent._alpha_optim = optim.Adam([agent._log_alpha], lr=lr_alpha)

    if beta_policy:
        apply_beta_policy(agent, mdp, actor_mu_params, actor_sigma_params, actor_optimizer)

    return agent

def build_datacom_sac(mdp, control_system, initial_replay_size, max_replay_size, batch_size, n_features_actor,
                      n_features_critic, n_features_constraint, learning_rate_actor, learning_rate_critic, activation,
                      accepted_risk, learning_rate_constraint, constraint_weight_decay,
                      slack_limit, atacom_lam, atacom_beta, use_cuda, tau, lr_alpha, target_entropy, atacom_dc, use_viability, violation_memory_ratio,
                      warmup_transitions, cost_budget, constr_aggregation, constr_aggregation_value_function, lr_delta, init_delta, delta_warmup_transitions, **kwargs):
    constraint_params = build_constraint(control_system, "gaussian",
                                         learning_rate_constraint, n_features_constraint, use_cuda)

    actor_mu_params, actor_sigma_params, actor_optimizer, critic_params, alg_params = \
        build_sac_params(mdp, n_features_actor, n_features_critic, learning_rate_actor, learning_rate_critic, activation,
                         use_cuda, tau, lr_alpha, target_entropy, warmup_transitions)

    if hasattr(mdp, "analytical_constraint"):
        alg_params["analytical_constraint"] = mdp.analytical_constraint

    agent = DatacomSAC(mdp_info=mdp.info, control_system=control_system, accepted_risk=accepted_risk,
                       actor_mu_params=actor_mu_params, actor_sigma_params=actor_sigma_params,
                       actor_optimizer=actor_optimizer, critic_params=critic_params, batch_size=batch_size,
                       initial_replay_size=initial_replay_size, max_replay_size=max_replay_size,
                       cost_budget=cost_budget, constraint_params=constraint_params, constr_aggregation=constr_aggregation, constr_aggregation_value_function=constr_aggregation_value_function, 
                       slack_limit=slack_limit, atacom_lam=atacom_lam, atacom_beta=atacom_beta, lr_delta=lr_delta, init_delta=init_delta, constraint_weight_decay=constraint_weight_decay,
                       delta_warmup_transitions=delta_warmup_transitions, atacom_dc=atacom_dc, use_viability=use_viability, n_learnable_constr=mdp.n_learnable_constr, violation_memory_ratio=violation_memory_ratio,
                       **alg_params)

    return agent


def build_cbf_sac(mdp, control_system, initial_replay_size, max_replay_size, batch_size, n_features_actor,
                  n_features_critic, n_features_constraint, learning_rate_actor, learning_rate_critic, activation,
                  learning_rate_constraint,
                  use_cuda, tau, lr_alpha, target_entropy,
                  warmup_transitions, **kwargs):
    constraint_params = dict(network=MLP,
                             optimizer={'class': optim.Adam,
                                        'params': {'lr': learning_rate_constraint}},
                             n_features=list(map(int, n_features_constraint.split(' '))),
                             input_shape=(control_system.dim_q + control_system.dim_x,),
                             output_shape=(1,),
                             use_cuda=use_cuda,
                             quiet=True,
                             loss=F.mse_loss,
                             activation='relu')

    actor_mu_params, actor_sigma_params, actor_optimizer, critic_params, alg_params = \
        build_sac_params(mdp, n_features_actor, n_features_critic, learning_rate_actor, learning_rate_critic, activation,
                         use_cuda, tau, lr_alpha, target_entropy, warmup_transitions)

    if hasattr(mdp, "analytical_constraint"):
        alg_params["analytical_constraint"] = mdp.analytical_constraint

    agent = CBFSAC(mdp_info=mdp.info, control_system=control_system,
                   actor_mu_params=actor_mu_params, actor_sigma_params=actor_sigma_params,
                   actor_optimizer=actor_optimizer, critic_params=critic_params, batch_size=batch_size,
                   initial_replay_size=initial_replay_size, max_replay_size=max_replay_size,
                   constraint_params=constraint_params,
                   **alg_params)

    return agent


def build_iqn_datacom_sac(mdp, control_system, initial_replay_size, max_replay_size, batch_size, n_features_actor,
                          n_features_critic, n_features_constraint, learning_rate_actor, learning_rate_critic, activation,
                          learning_rate_constraint, accepted_risk,
                          slack_limit, atacom_lam, atacom_beta, use_cuda, tau, lr_alpha, target_entropy, warmup_transitions,
                          quantile_embedding_dim, num_quantile_samples,
                          num_next_quantile_samples,
                          cost_budget, lr_delta, init_delta, delta_warmup_transitions, **kwargs):
    constraint_params = build_constraint(control_system, 'quantile',
                                         learning_rate_constraint, n_features_constraint, use_cuda)

    constraint_params['embedding_size'] = quantile_embedding_dim

    actor_mu_params, actor_sigma_params, actor_optimizer, critic_params, alg_params = \
        build_sac_params(mdp, n_features_actor, n_features_critic, learning_rate_actor, learning_rate_critic, activation,
                         use_cuda, tau, lr_alpha, target_entropy, warmup_transitions)

    agent = IQNAtacomSAC(mdp_info=mdp.info, control_system=control_system, accepted_risk=accepted_risk,
                         actor_mu_params=actor_mu_params, actor_sigma_params=actor_sigma_params,
                         actor_optimizer=actor_optimizer,
                         critic_params=critic_params, batch_size=batch_size, initial_replay_size=initial_replay_size,
                         max_replay_size=max_replay_size, cost_budget=cost_budget, constraint_params=constraint_params,
                         slack_limit=slack_limit, atacom_lam=atacom_lam, atacom_beta=atacom_beta,
                         lr_delta=lr_delta, init_delta=init_delta, delta_warmup_transitions=delta_warmup_transitions,
                         num_quantile_samples=num_quantile_samples, num_next_quantile_samples=num_next_quantile_samples,
                         **alg_params)
    return agent


def build_lag_sac(mdp, initial_replay_size, max_replay_size, batch_size, n_features_actor, n_features_critic,
                  learning_rate_actor, learning_rate_critic, learning_rate_constraint, activation, tau, lr_alpha, target_entropy,
                  warmup_transitions, use_cuda, lr_beta, cost_limit, damp_scale, **kwargs):
    actor_mu_params, actor_sigma_params, actor_optimizer, critic_params, alg_params = \
        build_sac_params(mdp, n_features_actor, n_features_critic, learning_rate_actor, learning_rate_critic, activation,
                         use_cuda, tau, lr_alpha, target_entropy, warmup_transitions)

    constraint_params = dict(network=SACCriticNetwork,
                             input_shape=(mdp.info.observation_space.shape[0] + mdp.info.action_space.shape[0],),
                             optimizer={'class': optim.Adam,
                                        'params': {'lr': learning_rate_constraint}},
                             loss=F.mse_loss,
                             n_features=list(map(int, n_features_critic.split(' '))),
                             output_shape=(1,),
                             use_cuda=use_cuda)

    agent = LagSAC(mdp.info, actor_mu_params, actor_sigma_params, actor_optimizer, critic_params, **alg_params,
                   constraint_params=constraint_params, initial_replay_size=initial_replay_size,
                   max_replay_size=max_replay_size, batch_size=batch_size,
                   lr_beta=lr_beta, cost_limit=cost_limit, damp_scale=damp_scale)

    return agent


def build_wcsac(mdp, initial_replay_size, max_replay_size, batch_size, n_features_actor, n_features_critic,
                learning_rate_actor, learning_rate_critic, learning_rate_constraint, activation, accepted_risk, tau, lr_alpha,
                target_entropy, warmup_transitions, lr_beta, cost_limit, damp_scale, constraint_type, **kwargs):
    use_cuda = False

    actor_mu_params, actor_sigma_params, actor_optimizer, critic_params, alg_params = \
        build_sac_params(mdp, n_features_actor, n_features_critic, learning_rate_actor, learning_rate_critic, activation, 
                         use_cuda, tau, lr_alpha, target_entropy, warmup_transitions)

    if constraint_type == "gaussian":
        net = GaussianConstraintQNetwork
    elif constraint_type == "quantile":
        net = ImplicitQuantileConstraint

    constraint_params = dict(network=net,
                             input_shape=(mdp.info.observation_space.shape[0] + mdp.info.action_space.shape[0],),
                             optimizer={'class': optim.Adam,
                                        'params': {'lr': learning_rate_constraint}},
                             n_features=list(map(int, n_features_critic.split(' '))),
                             output_shape=(1,),
                             use_cuda=use_cuda,
                             embedding_dim=n_features_critic,
                             num_cosines=n_features_critic)

    agent = WCSAC(mdp.info, actor_mu_params, actor_sigma_params, actor_optimizer, critic_params, **alg_params,
                  constraint_params=constraint_params, initial_replay_size=initial_replay_size,
                  max_replay_size=max_replay_size, batch_size=batch_size,
                  lr_beta=lr_beta, cost_limit=cost_limit, accepted_risk=accepted_risk, damp_scale=damp_scale,
                  constraint_type=constraint_type)

    return agent
