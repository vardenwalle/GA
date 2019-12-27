from genalg import GA

def function(x):
    z = x[0] + 2 * x[1] + 3 * x[2] + 4 * x[3]
    ans = 30
    print("{:.1f} {:+.1f}*2 {:+.1f}*3 {:+.1f}*4 = {:.1f}".format(x[0], x[1], x[2], x[3], z), "- Solved!" if z == ans else "")
    return -abs(ans-z)

ga = GA(function, bounds = (-100, 100, 1), num_genes=4, steps=30, stop_fitness=0, stagnation=3,
        population_limit=10, survive_coef = 0.25, productivity=4, default_step=1, mutagen="1_step", verbose=True)
result = ga.evolve()
print("Best solution:", result)


