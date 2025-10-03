from Random_search.Adaptive_random_search import AdaptiveRandomSearch as ARS
from Random_search.Random_Search import RandomSearch as RS

# Sphere function minimization
def sphere(v):
    return sum(x**2 for x in v)


# Random search sample usage

bounds = [(-5, 5), (-5, 5)]
res = RS.random_search(sphere, bounds, iterations=1000, seed=42)
print("best_x:", res["best_x"])
print("best_score:", res["best_score"])


print("\n\n_____________________\n\n")

# Adaptive random search sample usage

bounds = [(-5, 5), (-5, 5)]
ars = ARS(sphere, bounds, step_size=0.1, iterations=2000, maximize=False, seed=42)
result = ars.run()

print("Best solution:", result["best_x"])
print("Best score:", result["best_score"])

