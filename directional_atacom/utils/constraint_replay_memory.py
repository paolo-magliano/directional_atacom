import numpy as np

from mushroom_rl.core import Serializable

from mushroom_rl.utils.replay_memory import ReplayMemory



class SafeReplayMemory(ReplayMemory):
    def __init__(self, initial_size, max_size):
        super().__init__(initial_size, max_size)

        self._add_save_attr(
            _cost='pickle!',
        )

    def add(self, dataset):
        """
        Add elements to the replay memory.

        Args:
            dataset (list): list of elements to add to the replay memory;
        """

        for el in dataset:
            self._states[self._idx] = el[0]
            self._actions[self._idx] = el[1]
            self._rewards[self._idx] = el[2]
            self._next_states[self._idx] = el[3]
            self._cost[self._idx] = el[4]
            self._absorbing[self._idx] = el[5]
            self._last[self._idx] = el[6]

            self._idx += 1
            if self._idx == self._max_size:
                self._full = True
                self._idx = 0

    def get(self, n_samples):
        """
        Returns the provided number of states from the replay memory.
        Args:
            n_samples (int): the number of samples to return.
        Returns:
            The requested number of samples.
        """
        s = list()
        a = list()
        r = list()
        ss = list()
        c = list()
        ab = list()
        last = list()
        for i in np.random.randint(self.size, size=n_samples):
            s.append(np.array(self._states[i]))
            a.append(self._actions[i])
            r.append(self._rewards[i])
            ss.append(np.array(self._next_states[i]))
            c.append(self._cost[i])
            ab.append(self._absorbing[i])
            last.append(self._last[i])

        return np.array(s), np.array(a), np.array(r), np.array(ss), \
            np.array(c), np.array(ab), np.array(last)

    def reset(self):
        super(SafeReplayMemory, self).reset()

        self._cost = [None for _ in range(self._max_size)]
