import pandas as pd
from functools import partial
from timeit import Timer
from tsm_solver.solver import TsmProblemSolver
from bench.benchmark_utils import generate_matrix

if __name__ == '__main__':
    solver = TsmProblemSolver()

    results = []

    for i in range(2, 8):
        n_iteration = 5
        matrix = generate_matrix(i)

        t_brute = Timer(partial(solver.solve, matrix, 'brute_force'))
        t_greedy = Timer(partial(solver.solve, matrix, 'greedy'))
        t_christ = Timer(partial(solver.solve, matrix, 'christofides'))

        time_brute = t_brute.timeit(number=n_iteration)
        time_greedy = t_greedy.timeit(number=n_iteration)
        time_christ = t_christ.timeit(number=n_iteration)

        results.append((i,
                        n_iteration,
                        time_brute,
                        time_greedy,
                        time_christ))

        print('Brute force mean time:', time_brute)
        print('Greedy mean time:', time_greedy)
        print('Christofides mean time:', time_christ)

    for i in range(8, 151):
        n_iteration = 10
        matrix = generate_matrix(i)

        t_greedy = Timer(partial(solver.solve, matrix, 'greedy'))
        t_christ = Timer(partial(solver.solve, matrix, 'christofides'))

        time_greedy = t_greedy.timeit(number=n_iteration)
        time_christ = t_christ.timeit(number=n_iteration)

        results.append((i,
                        n_iteration,
                        None,
                        time_greedy,
                        time_christ))

        print('Greedy mean time:', time_greedy)
        print('Christofides mean time:', time_christ)

    df = pd.DataFrame(data=results, columns=['N', 'Repeats', 'Bruteforce', 'Greedy', 'Christofides'])
    df.to_csv('./results.csv')
