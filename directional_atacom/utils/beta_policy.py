import numpy as np
import torch.nn

from mushroom_rl.policy import Policy
from mushroom_rl.utils.torch import to_float_tensor
from mushroom_rl.utils.parameters import to_parameter
from torch.distributions import TransformedDistribution, Beta, AffineTransform
from itertools import chain

class BetaPolicy(Policy):
    """
    Class used to implement the policy used by the Soft Actor-Critic algorithm.
    The policy is a Gaussian policy squashed by a tanh. This class implements the compute_action_and_log_prob and the
    compute_action_and_log_prob_t methods, that are fundamental for the internals calculations of the SAC algorithm.

    """

    def __init__(self, alpha_approximator, beta_approximator, min_a, max_a):
        """
        Constructor.

        Args:
            alpha_approximator (Regressor): a regressor computing alpha_parameter;
            beta_approximator (Regressor): a regressor computing the beta_parameter;
            min_a (np.ndarray): a vector specifying the minimum action value for each component;
            max_a (np.ndarray): a vector specifying the maximum action value for each component.
            params_min ([float, Parameter]): min value of the parameters;
            params_max ([float, Parameter]): max value of the parameters.

        """
        self._alpha_approximator = alpha_approximator
        self._beta_approximator = beta_approximator

        self._delta_a = to_float_tensor(.5 * (max_a - min_a), self.use_cuda)
        self._central_a = to_float_tensor(.5 * (max_a + min_a), self.use_cuda)

        self._eps_log_prob = 1e-6

        use_cuda = self._alpha_approximator.model.use_cuda

        if use_cuda:
            self._delta_a = self._delta_a.cuda()
            self._central_a = self._central_a.cuda()

        self._add_save_attr(
            _alpha_approximator='mushroom',
            _beta_approximator='mushroom',
            _delta_a='torch',
            _central_a='torch',
            _eps_log_prob='primitive',
        )

    def __call__(self, state, action):
        raise NotImplementedError

    def draw_action(self, state):
        return self.compute_action_and_log_prob_t(state, compute_log_prob=False).detach().cpu().numpy()

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

    def compute_action_and_log_prob_t(self, state, compute_log_prob=True):
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
        a = dist.rsample()

        if compute_log_prob:
            a = torch.clamp(a, min=self._central_a - self._delta_a + 1e-6, max=self._central_a + self._delta_a - 1e-6)
            log_prob = dist.log_prob(a).sum(dim=1)
            return a, log_prob
        else:
            return a

    def distribution(self, state):
        """
        Compute the policy distribution in the given states.

        Args:
            state (np.ndarray): the set of states where the distribution is computed.

        Returns:
            The torch distribution for the provided states.

        """
        alpha = self._alpha_approximator.predict(state, output_tensor=True)
        beta = self._beta_approximator.predict(state, output_tensor=True)

        alpha = torch.nn.functional.softplus(alpha) + 1.
        beta = torch.nn.functional.softplus(beta) + 1.

        dist = TransformedDistribution(Beta(alpha, beta),
                                       [AffineTransform(self._central_a - self._delta_a, 2 * self._delta_a)])
        return dist

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

    def compute_log_prob(self, state, action):
        dist = self.distribution(state)
        clamped_action = torch.clamp(action, self._central_a - self._delta_a + self._eps_log_prob,
                                     self._central_a + self._delta_a - self._eps_log_prob)
        return dist.log_prob(clamped_action)

    def set_weights(self, weights):
        """
        Setter.

        Args:
            weights (np.ndarray): the vector of the new weights to be used by the policy.

        """
        alpha_weights = weights[:self._alpha_approximator.weights_size]
        beta_weights = weights[self._beta_approximator.weights_size:]

        self._alpha_approximator.set_weights(alpha_weights)
        self._beta_approximator.set_weights(beta_weights)

    def get_weights(self):
        """
        Getter.

        Returns:
             The current policy weights.

        """
        alpha_weights = self._alpha_approximator.get_weights()
        beta_weights = self._beta_approximator.get_weights()

        return np.concatenate([alpha_weights, beta_weights])

    @property
    def use_cuda(self):
        """
        True if the policy is using cuda_tensors.
        """
        return self._alpha_approximator.model.use_cuda

    def parameters(self):
        """
        Returns the trainable policy parameters, as expected by torch optimizers.

        Returns:
            List of parameters to be optimized.

        """
        return chain(self._alpha_approximator.model.network.parameters(),
                     self._beta_approximator.model.network.parameters())