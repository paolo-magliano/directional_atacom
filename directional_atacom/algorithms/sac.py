import numpy as np

from mushroom_rl.algorithms.actor_critic.deep_actor_critic import SAC as BaseSAC

from directional_atacom.utils.action_filter import ActionFilter


class SAC(BaseSAC):
    """
    SAC with an optional action filter, applied in ``preprocess_action`` so that the
    dataset keeps the raw policy sample while the environment receives the filtered
    action. With ``action_filter_ratio=None`` it behaves exactly like the base SAC.

    Do not override ``fit`` here: under ExtCore the dataset carries no cost column.
    Subclasses running under SafeCore (e.g. AtacomSAC) drop it themselves.

    """
    def __init__(self, mdp_info, actor_mu_params, actor_sigma_params, actor_optimizer, critic_params, batch_size,
                 initial_replay_size, max_replay_size, warmup_transitions, tau, lr_alpha, use_log_alpha_loss=False,
                 log_std_min=-20, log_std_max=2, target_entropy=None, critic_fit_params=None,
                 action_filter_ratio=None, save_prev_action=None):
        super().__init__(mdp_info, actor_mu_params, actor_sigma_params, actor_optimizer, critic_params, batch_size,
                         initial_replay_size, max_replay_size, warmup_transitions, tau, lr_alpha, use_log_alpha_loss,
                         log_std_min, log_std_max, target_entropy, critic_fit_params)

        self.action_filter = ActionFilter(action_filter_ratio, mdp_info.action_space.shape, save_prev_action)

    def preprocess_action(self, state, action, next_cost):
        action = np.clip(action, self.mdp_info.action_space.low, self.mdp_info.action_space.high)

        return self.action_filter(action)

    def episode_start(self):
        super().episode_start()
        self.action_filter.reset()
