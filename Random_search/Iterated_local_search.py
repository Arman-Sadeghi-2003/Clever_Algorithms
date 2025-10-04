import random
from typing import Callable, List, Tuple, Optional, Union

Number = Union[int, float]
Vector = List[Number]

class IteratedLocalSearch:
    """
    Iterated Local Search (ILS) optimizer.
    Combines local search (hill climbing) with perturbations to escape local optima.
    """

    def __init__(self,
                 objective: Callable[[Vector], float],
                 bounds: List[Tuple[Number, Number]],
                 step_size: float = 0.1,
                 iterations: int = 1000,
                 perturb_strength: float = 0.5,
                 maximize: bool = False,
                 seed: Optional[int] = None):
        """
        Args:
            objective: function f(x) to minimize/maximize.
            bounds: list of (low, high) for each dimension.
            step_size: step size used in local search perturbations.
            iterations: total number of iterations (local searches + perturbations).
            perturb_strength: scale of "big jump" when perturbing best solution.
            maximize: if True → maximize; else minimize.
            seed: optional random seed.
        """
        self.objective = objective
        self.bounds = bounds
        self.step_size = step_size
        self.iterations = iterations
        self.perturb_strength = perturb_strength
        self.maximize = maximize
        self.rng = random.Random(seed)
        self.dim = len(bounds)

    def _random_vector(self) -> Vector:
        """Random sample in bounds."""
        return [self.rng.uniform(lo, hi) for lo, hi in self.bounds]

    def _neighbor(self, vector: Vector, step: float) -> Vector:
        """Generate a neighbor by small Gaussian perturbation."""
        candidate = []
        for (lo, hi), v in zip(self.bounds, vector):
            span = hi - lo
            perturbed = v + self.rng.gauss(0, step * span)
            perturbed = max(lo, min(hi, perturbed))  # clip to bounds
            candidate.append(perturbed)
        return candidate

    def _hill_climb(self, start: Vector, max_iters: int = 100) -> Tuple[Vector, float]:
        """Perform simple hill climbing from a start vector."""
        current = start[:]
        current_score = self.objective(current)

        for _ in range(max_iters):
            candidate = self._neighbor(current, self.step_size)
            candidate_score = self.objective(candidate)

            improved = candidate_score > current_score if self.maximize else candidate_score < current_score
            if improved:
                current, current_score = candidate, candidate_score

        return current, current_score

    def _perturb(self, vector: Vector) -> Vector:
        """Make a big jump (perturbation) to escape local optimum."""
        return self._neighbor(vector, self.perturb_strength)

    def run(self) -> dict:
        """Execute Iterated Local Search."""
        # initial solution
        current = self._random_vector()
        current, current_score = self._hill_climb(current)

        best, best_score = current[:], current_score
        history = [(0, best_score)]

        for it in range(1, self.iterations + 1):
            # perturb best
            perturbed = self._perturb(best)
            candidate, candidate_score = self._hill_climb(perturbed)

            improved = candidate_score > best_score if self.maximize else candidate_score < best_score
            if improved:
                best, best_score = candidate[:], candidate_score

            history.append((it, best_score))

        return {
            "best_x": best,
            "best_score": best_score,
            "history": history,
            "iterations": self.iterations
        }
