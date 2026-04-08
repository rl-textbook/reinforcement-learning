"""Replay buffer for DQN and related value-based agents."""

from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.buf = [None] * int(capacity)
        self.cap = int(capacity)
        self.idx = 0
        self.full = False

    def push(self, s, a, r, s_next, done):
        # Store one transition ( S_t, A_t, R_{t+1}, S_{t+1}, done )
        self.buf[self.idx] = (s, a, r, s_next, done)
        self.idx = (self.idx + 1) % self.cap
        if self.idx == 0:
            self.full = True # overwrite the oldest transition

    def __len__(self):
        return self.cap if self.full else self.idx

    def sample(self, batch_size: int):
        # Uniform random indices approximate i.i.d. training data
        n = len(self)
        if batch_size > n:
            raise ValueError(f"Cannot sample batch_size={batch_size} from buffer of size {n}")
        idxs = np.random.choice(n, size=batch_size, replace=False)
        batch = [self.buf[i] for i in idxs]
        s, a, r, s_next, d = zip(*batch)
        return (
            np.asarray(s, dtype=np.float32),
            np.asarray(a, dtype=np.int64),
            np.asarray(r, dtype=np.float32),
            np.asarray(s_next, dtype=np.float32),
            np.asarray(d, dtype=np.float32),
        )
