import random
import math
from typing import Callable, Dict, List, Tuple, Sequence, Union, Optional

Number = Union[int, float]
Vector = Sequence[Number]

def _sample_from_bounds(bounds: Sequence[Tuple[Number, Number]],
                        integer: Sequence[bool],
                        rng: random.Random) -> List[Number]:
    """Sample one candidate within bounds. `integer[i]` says if coordinate i is integer."""
    candidate = []
    for (lo, hi), is_int in zip(bounds, integer):
        if is_int:
            # randint requires int bounds; allow floats too (rounding)
            candidate.append(rng.randint(math.ceil(lo), math.floor(hi)))
        else:
            candidate.append(rng.uniform(lo, hi))
    return candidate

def random_search(objective: Callable[[Vector], float],
                  bounds: Sequence[Tuple[Number, Number]],
                  iterations: int = 1000,
                  integer: Optional[Sequence[bool]] = None,
                  maximize: bool = False,
                  seed: Optional[int] = None,
                  return_history: bool = False,
                  verbose: bool = False) -> Dict:
    """
    Perform a random search to (approximately) optimize `objective`.

    Args:
      objective: function mapping vector -> scalar (the score).
                 For minimization, objective smaller is better. For maximization set maximize=True.
      bounds: sequence of (low, high) pairs for each dimension.
      iterations: number of random candidates to try.
      integer: optional bool sequence same length as bounds. True => sample integers for that dim.
               If None, all dims treated as continuous floats.
      maximize: if True, tries to maximize objective; otherwise minimize.
      seed: optional random seed for reproducibility.
      return_history: if True, returns list of best scores after each improvement attempt.
      verbose: if True, prints progress info every 10% of iterations.

    Returns:
      dict with keys:
        - "best_x": best found vector
        - "best_score": best objective value (already respecting minimize/maximize)
        - "history": list of tuples (iteration_index, best_score) if return_history True
        - "evaluations": total iterations performed
    """
    rng = random.Random(seed)
    dim = len(bounds)
    if integer is None:
        integer = [False] * dim
    else:
        if len(integer) != dim:
            raise ValueError("`integer` length must equal number of bounds")

    # Initialize with one sample
    best_x = _sample_from_bounds(bounds, integer, rng)
    best_score_raw = objective(best_x)
    # Convert to 'score' where higher is better internally
    best_score = best_score_raw if maximize else -best_score_raw

    history = []
    if return_history:
        history.append((0, best_score_raw))

    report_every = max(1, iterations // 10)
    for it in range(1, iterations + 1):
        x = _sample_from_bounds(bounds, integer, rng)
        raw = objective(x)
        score = raw if maximize else -raw

        if score > best_score:
            best_score = score
            best_score_raw = raw
            best_x = x
            if return_history:
                history.append((it, best_score_raw))

        if verbose and it % report_every == 0:
            print(f"[random_search] iter {it}/{iterations}, best_score={best_score_raw}")

    result = {
        "best_x": best_x,
        "best_score": best_score_raw,
        "evaluations": iterations
    }
    if return_history:
        result["history"] = history
    return result


# minimize the 2D sphere function f(x)=x0^2 + x1^2
def sphere(v): 
    return v[0]**2 + v[1]**2

bounds = [(-5, 5), (-5, 5)]
res = random_search(sphere, bounds, iterations=2000, seed=42, maximize=False, return_history=True)
print("best_x:", res["best_x"])
print("best_score:", res["best_score"])