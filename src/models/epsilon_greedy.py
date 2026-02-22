import random
from typing import List


class EpsilonGreedy:
    """Epsilon-Greedy bandit where each arm is a recommender model.

    With probability epsilon a random arm is chosen (exploration);
    otherwise the arm with the highest observed average reward is
    chosen (exploitation).
    """

    def __init__(self, n_arms: int, epsilon: float = 0.1, seed: int = 42):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts: List[int] = [0] * n_arms        # pulls per arm
        self.rewards: List[float] = [0.0] * n_arms    # cumulative reward per arm
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    def select_arm(self) -> int:
        """Return the index of the arm to pull."""
        if self._rng.random() < self.epsilon:
            return self._rng.randrange(self.n_arms)
        # exploit: pick arm with best average (break ties randomly)
        averages = [
            self.rewards[a] / self.counts[a] if self.counts[a] > 0 else 0.0
            for a in range(self.n_arms)
        ]
        max_avg = max(averages)
        best = [a for a, avg in enumerate(averages) if avg == max_avg]
        return self._rng.choice(best)

    # ------------------------------------------------------------------
    def update(self, arm: int, reward: float) -> None:
        """Record a reward observation for *arm*."""
        self.counts[arm] += 1
        self.rewards[arm] += reward
