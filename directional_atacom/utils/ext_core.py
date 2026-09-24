import numpy as np
from mushroom_rl.core import Core


class ExtCore(Core):
    """
    Core extended with an action preprocessing step, the counterpart of the state
    preprocessing already provided by mushroom. An agent may define
    ``preprocess_action(state, action, cost)`` to transform the action sent to the
    environment. The action stored in the dataset is the one the agent drew, so the
    learning update always sees the raw policy sample.

    """
    def _preprocess_action(self, action, cost=None):
        """
        Method to apply the agent's action preprocessing, if any.

        Args:
            action (np.ndarray): the action drawn by the agent;
            cost: the cost of the previous step, when the core tracks it.

        Returns:
             The action to send to the environment.

        """
        if hasattr(self.agent, 'preprocess_action') and callable(self.agent.preprocess_action):
            return self.agent.preprocess_action(self._state[np.newaxis], action, cost)

        return action

    def _step(self, render, record):
        """
        Single step.

        Args:
            render (bool): whether to render or not.

        Returns:
            A tuple containing the previous state, the action sampled by the agent, the reward obtained, the reached
            state, the absorbing flag of the reached state and the last step flag.

        """
        action = self.agent.draw_action(self._state)
        next_state, reward, absorbing, step_info = self.mdp.step(self._preprocess_action(action))

        self._episode_steps += 1

        if render:
            frame = self.mdp.render(record)

            if record:
                self._record(frame)

        last = not(
            self._episode_steps < self.mdp.info.horizon and not absorbing)

        state = self._state
        next_state = self._preprocess(next_state.copy())
        self._state = next_state

        return (state, action, reward, next_state, absorbing, last), step_info
