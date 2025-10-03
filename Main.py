from Random_search.Adaptive_random_search import AdaptiveRandomSearch as ARS
from Random_search.Random_Search import RandomSearch as RS

# Sphere function minimization
def sphere(v):
    return sum(x**2 for x in v)

bounds = [(-5, 5), (-5, 5)]
ars = ARS(sphere, bounds, step_size=0.1, iterations=2000, maximize=False, seed=42)
result = ars.run()

print("Best solution:", result["best_x"])
print("Best score:", result["best_score"])



bounds = [(-5, 5), (-5, 5)]
res = RS(sphere, bounds, iterations=2000, seed=42, maximize=False, return_history=True)
print("best_x:", res["best_x"])
print("best_score:", res["best_score"])