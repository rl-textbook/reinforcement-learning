"""
Replay Buffer for experience replay in DQN and related algorithms.

See Chapter 4, Section on Deep Q-Networks for the role of experience
replay in stabilizing training.
"""

import numpy as np


# --- Replay Buffer for DQN ---
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf = [None] * capacity
        self.cap = capacity
        self.idx = 0
        self.full = False
    def push(self, s, a, r, s_next, done):
        self.buf[self.idx] = (s, a, r, s_next, done)
        self.idx = (self.idx + 1) % self.cap
        if self.idx == 0:
            self.full = True
    def __len__(self):
        return self.cap if self.full else self.idx
    def sample(self, batch_size: int):
        n = len(self)
        idxs = np.random.choice(n, size=batch_size, replace=False)
        batch = [self.buf[i] for i in idxs]
        s, a, r, s_next, d = zip(*batch)
        return np.array(s, dtype=np.float32), np.array(a, dtype=np.int64), \
               np.array(r, dtype=np.float32), np.array(s_next, dtype=np.float32), \
               np.array(d, dtype=np.float32)
