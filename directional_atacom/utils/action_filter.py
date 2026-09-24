import numpy as np


class ActionFilter:
    """
    Exponential moving-average filter on the commanded action, owned by the agent.

    It is meant to be applied in the agent's ``preprocess_action``, i.e. after the
    core has captured the raw policy sample for the dataset and before the action
    reaches the environment (and, for ATACOM, the projection). This keeps the
    learning update on the raw sample while the plant sees a smoothed command.

    ``save_prev_action`` lets the environment expose the previous filtered action
    in the observation, which keeps the filtered plant Markov.

    """
    def __init__(self, ratio, action_shape, save_prev_action=None):
        self.ratio = ratio
        self.action_shape = action_shape
        self.save_prev_action = save_prev_action
        self.reset()

    def reset(self):
        self.prev_action = np.zeros(self.action_shape)
        self._publish()

    def __call__(self, action):
        if self.ratio:
            action = (1 - self.ratio) * self.prev_action + self.ratio * action
            self.prev_action = action.copy()
            self._publish()

        return action

    def _publish(self):
        if self.save_prev_action is not None:
            self.save_prev_action(self.prev_action)
