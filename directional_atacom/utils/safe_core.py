from directional_atacom.utils.ext_core import ExtCore


class SafeCore(ExtCore):
    """
    ExtCore for safe algorithms: the environment returns a cost, which is tracked
    across steps, passed to the agent's action preprocessing and stored in the
    dataset.

    """
    def reset(self, initial_states=None):
        super(SafeCore, self).reset(initial_states)
        self._cost = 0

    def _step(self, render, record):
        """
        Single step.

        Args:
            render (bool): whether to render or not.

        Returns:
            A tuple containing the previous state, the action sampled by the agent, the reward obtained, the reached
            state, the cost obtained, the absorbing flag of the reached state and the last step flag.

        """
        action = self.agent.draw_action(self._state)
        next_state, reward, cost, absorbing, step_info = self.mdp.step(self._preprocess_action(action, self._cost))

        self._episode_steps += 1

        if render:
            frame = self.mdp.render(record)

            if record:
                self._record(frame)

        last = not (
            self._episode_steps < self.mdp.info.horizon and not absorbing)

        state = self._state
        next_state = self._preprocess(next_state.copy())
        self._state = next_state

        self._cost = cost

        return (state, action, reward, next_state, cost, absorbing, last), step_info
