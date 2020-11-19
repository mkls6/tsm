import numpy as np
from functools import partial
from timeit import Timer
from tsm_solver.solver import TsmProblemSolver

if __name__ == '__main__':
    solver = TsmProblemSolver()
    matrix = np.array([[0, 5, 5.83, 2],
                       [5, 0, 3, 6.24],
                       [5.83, 3, 0, 5.44],
                       [2, 6.24, 5.44, 0]])

    solver.solve(matrix, 'christofides')

    t_brute = Timer(partial(solver.solve, matrix, 'brute_force'))
    t_greedy = Timer(partial(solver.solve, matrix, 'greedy'))

    print('Brute force mean time:', t_brute.timeit(number=100000))
    print('Greedy mean time:', t_greedy.timeit(number=100000))
