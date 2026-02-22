import random
from typing import List


class ThompsonSampling:
    """Thompson Sampling bandit with Beta-Bernoulli conjugate model.

    Each arm maintains a Beta(alpha, beta) posterior.  At each step a
    sample is drawn from every arm's posterior and the arm with the
    highest sample is selected.  After observing a binary reward the
    posterior is updated: alpha += reward, beta += (1 - reward).
    """

    def __init__(self, n_arms: int, seed: int = 42):
        self.n_arms = n_arms
        self.alpha: List[float] = [1.0] * n_arms   # successes + 1 (prior)
        self.beta: List[float] = [1.0] * n_arms    # failures  + 1 (prior)
        self.counts: List[int] = [0] * n_arms
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    def select_arm(self) -> int:
        """Sample from each arm's Beta posterior and return the best."""
        samples = [
            self._rng.betavariate(self.alpha[a], self.beta[a])
            for a in range(self.n_arms)
        ]
        return max(range(self.n_arms), key=lambda a: samples[a])

    # ------------------------------------------------------------------
    def update(self, arm: int, reward: float) -> None:
        """Update the posterior for *arm* given a binary reward (0 or 1)."""
        self.counts[arm] += 1
        self.alpha[arm] += reward
        self.beta[arm] += 1.0 - reward
