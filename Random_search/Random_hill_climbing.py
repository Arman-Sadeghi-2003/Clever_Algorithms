import random
import math
from typing import Callable, List, Tuple, Optional, Union

Number = Union[int, float]
Vector = List[Number]

class RandomHillClimbing:
    """
    Random Hill Climbing (RHC) optimizer.

    Works for continuous optimization problems.
    """

    def __init__(self,
                 objective: Callable[[Vector], float],
                 bounds: List[Tuple[Number, Number]],
                 step_size: float = 0.1,
                 iterations: int = 1000,
                 maximize: bool = False,
                 seed: Optional[int] = None):
        """
        Args:
            objective: function f(x) to minimize/maximize.
            bounds: list of (low, high) for each dimension.
            step_size: size of neighbor perturbation (fraction of bound range).
            iterations: number of iterations to run.
            maximize: if True → maximize; else minimize.
            seed: optional random seed.
        """
        self.objective = objective
        self.bounds = bounds
        self.step_size = step_size
        self.iterations = iterations
        self.maximize = maximize
        self.rng = random.Random(seed)
        self.dim = len(bounds)

    def _random_vector(self) -> Vector:
        """Random sample in bounds."""
        return [self.rng.uniform(lo, hi) for lo, hi in self.bounds]

    def _perturb(self, vector: Vector) -> Vector:
        """Generate a neighbor by perturbing each dimension."""
        candidate = []
        for (lo, hi), v in zip(self.bounds, vector):
            span = hi - lo
            perturbed = v + self.rng.gauss(0, self.step_size * span)
            perturbed = max(lo, min(hi, perturbed))  # clip to bounds
            candidate.append(perturbed)
        return candidate

    def run(self) -> dict:
        """Run the hill climbing search."""
        current = self._random_vector()
        current_score = self.objective(current)
        best = current[:]
        best_score = current_score
        history = [(0, best_score)]

        for it in range(1, self.iterations + 1):
            candidate = self._perturb(current)
            candidate_score = self.objective(candidate)

            improved = candidate_score > current_score if self.maximize else candidate_score < current_score

            if improved:
                current, current_score = candidate, candidate_score
                if (self.maximize and current_score > best_score) or (not self.maximize and current_score < best_score):
                    best, best_score = current[:], current_score

            history.append((it, best_score))

        return {
            "best_x": best,
            "best_score": best_score,
            "history": history,
            "iterations": self.iterations
        }
