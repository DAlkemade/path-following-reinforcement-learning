import random
from collections import deque, namedtuple


class Memory(object):
    """Store robot runs.

    Uses a deque with a fixed length, meaning that it only keeps the last N entries, i.e. it will forget old experiences.
    Adapted from
    https://github.com/shakedzy/notebooks/blob/master/q_learning_and_dqn/Q%20Learning%20and%20Deep%20Q%20Network.ipynb
    """

    def __init__(self, buffer_size):
        self.memory = deque(maxlen=buffer_size)

    @property
    def size(self):
        return len(self.memory)

    def append(self, item):
        """Append item to memory."""
        self.memory.append(item)

    def sample(self, size):
        """Sample elements from memory."""
        return random.sample(self.memory, size)

    def all_entries(self):
        """Return all entries, floored at the nearest 1000.

        This is useful to increase the batch size less often, so
        that the tensorflow graph does not have to be recreated every time for a slightly higher batch size.
        """
        batch_size = self.size - (self.size % 1000) if self.size >= 1000 else self.size
        return list(self.sample(batch_size))


Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])
