import random
import math
from typing import Callable, Dict, List, Tuple, Sequence, Union, Optional

class RandomSearch:
    Number = Union[int, float]
    Vector = Sequence[Number]

    @staticmethod
    def _sample_from_bounds(bounds: Sequence[Tuple[Number, Number]],
                            integer: Sequence[bool],
                            rng: random.Random) -> List[Number]:
        candidate = []
        for (lo, hi), is_int in zip(bounds, integer):
            if is_int:
                candidate.append(rng.randint(math.ceil(lo), math.floor(hi)))
            else:
                candidate.append(rng.uniform(lo, hi))
        return candidate

    @staticmethod
    def random_search(objective: Callable[[Vector], float],
                      bounds: Sequence[Tuple[Number, Number]],
                      iterations: int = 1000,
                      integer: Optional[Sequence[bool]] = None,
                      maximize: bool = False,
                      seed: Optional[int] = None,
                      return_history: bool = False,
                      verbose: bool = False) -> Dict:
        rng = random.Random(seed)
        dim = len(bounds)
        if integer is None:
            integer = [False] * dim
        elif len(integer) != dim:
            raise ValueError("`integer` length must equal number of bounds")

        best_x = RandomSearch._sample_from_bounds(bounds, integer, rng)
        best_score_raw = objective(best_x)
        best_score = best_score_raw if maximize else -best_score_raw

        history = []
        if return_history:
            history.append((0, best_score_raw))

        report_every = max(1, iterations // 10)
        for it in range(1, iterations + 1):
            x = RandomSearch._sample_from_bounds(bounds, integer, rng)
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
