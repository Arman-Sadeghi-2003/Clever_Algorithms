from Random_search.Adaptive_random_search import AdaptiveRandomSearch as ARS
from Random_search.Random_Search import RandomSearch as RS
from Random_search.Iterated_local_search import IteratedLocalSearch as ILS

# Sphere function minimization
def sphere(v):
    return sum(x**2 for x in v)


# Random search sample usage

print("RS:")

bounds = [(-5, 5), (-5, 5)]
res = RS.random_search(sphere, bounds, iterations=1000, seed=42)
print("best_x:", res["best_x"])
print("best_score:", res["best_score"])


print("\n\n_____________________\n\n")

# Adaptive random search sample usage

print("ARS:")

bounds = [(-5, 5), (-5, 5)]
ars = ARS(sphere, bounds, step_size=0.1, iterations=2000, maximize=False, seed=42)
result = ars.run()

print("Best solution:", result["best_x"])
print("Best score:", result["best_score"])


print("\n\n_____________________\n\n")

# Iterated local search usage sample

print("ILS:")

bounds = [(-5, 5), (-5, 5)]

ils = ILS(sphere, bounds,
          step_size=0.05,
          perturb_strength=0.5,
          iterations=1000,
          maximize=False,
          seed=42)

result = ils.run()

print("Best solution:", result["best_x"])
print("Best score:", result["best_score"])