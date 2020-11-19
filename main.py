import pandas as pd
from tqdm import tqdm
from functools import partial
from pytsp.christofides_tsp import christofides_tsp as c_tsp
from timeit import Timer
from tsm_solver.solver import TsmProblemSolver
from bench.benchmark_utils import generate_matrix

if __name__ == '__main__':
    solver = TsmProblemSolver()

    results = []
    # df = pd.DataFrame(columns=['N', 'Greedy T', 'Christofides T'])
    #
    # m = generate_matrix(5000)
    # # full, c1 = solver.solve(m, "brute_force")
    # greed, c2 = solver.solve(m, "greedy")
    # christ, c3 = solver.solve(m, "christofides")
    #
    # t_greedy = Timer(partial(solver.solve, m, 'greedy'))
    # t_christ = Timer(partial(solver.solve, m, 'christofides'))
    #
    # time_greedy = t_greedy.timeit(number=1) / 1
    # time_christ = t_christ.timeit(number=1) / 1
    #
    # tmp = {"N": 5000, "Greedy T": time_greedy, "Christofides T": time_christ}
    # df = df.append(tmp, ignore_index=True)
    #
    # df.to_csv('./del.csv', index=False)
    # exit(0)

    for i in range(2, 11):
        n_iteration = 5
        matrix = generate_matrix(i)

        t_brute = Timer(partial(solver.solve, matrix, 'brute_force'))
        t_greedy = Timer(partial(solver.solve, matrix, 'greedy'))
        t_christ = Timer(partial(solver.solve, matrix, 'christofides'))

        time_brute = t_brute.timeit(number=n_iteration) / n_iteration
        time_greedy = t_greedy.timeit(number=n_iteration) / n_iteration
        time_christ = t_christ.timeit(number=n_iteration) / n_iteration

        results.append((i,
                        n_iteration,
                        time_brute,
                        time_greedy,
                        time_christ))

        # print('Brute force mean time:', time_brute)
        # print('Greedy mean time:', time_greedy)
        # print('Christofides mean time:', time_christ)

    for i in tqdm(range(11, 2000, 100)):
        n_iteration = 10
        matrix = generate_matrix(i)

        t_greedy = Timer(partial(solver.solve, matrix, 'greedy'))
        t_christ = Timer(partial(solver.solve, matrix, 'christofides'))
        t_ctsp = Timer(partial(c_tsp, matrix, 0))

        time_greedy = t_greedy.timeit(number=n_iteration) / n_iteration
        time_christ = t_christ.timeit(number=n_iteration) / n_iteration
        time_ctsp = t_ctsp.timeit(number=n_iteration) / n_iteration

        results.append((i,
                        n_iteration,
                        None,
                        time_greedy,
                        time_christ))

        # print('Greedy mean time:', time_greedy)
        # print('Christofides mean time:', time_christ)

    df = pd.DataFrame(data=results, columns=['N', 'Repeats', 'Bruteforce', 'Greedy', 'Christofides'])
    df.to_csv('./results.csv', index=False)
