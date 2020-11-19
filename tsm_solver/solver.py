import numpy as np
from typing import Tuple, List
from itertools import permutations
from random import randint
from math import inf

GLOBAL_SHIT = []


def find_euler_cycle_rec(routes: np.ndarray, start_node: int) -> np.ndarray:
    for i, node in enumerate(routes[start_node]):
        if routes[start_node][i] >= 0.1:
            routes[start_node][i] -= 1
            find_euler_cycle_rec(routes, i)
    GLOBAL_SHIT.append(start_node)


def find_euler_cycle(routes: np.array) -> List:
    """
    Return euler chain from graph

    :param routes: adjacent matrix
    :return: 1d-array of vertices
    """
    st = [0]
    n = len(routes)

    res = []
    while len(st) > 0:
        v = st[-1]

        i = 0
        while i < n:
            if routes[v][i]:
                break
            i += 1

        if i == n:
            res.append(v)
            st.pop()
        else:
            routes[v][i] -= 1
            routes[i][v] -= 1
            st.append(i)

    return res


def prims_mst(routes: np.array) -> np.array:
    """
    Implements minimum spanning tree finder using Prim's algorithm
    :param routes: np.array matrix of edge weights
    :return: adjacent matrix
    """
    n = routes.shape[0]

    used = n * [False]
    min_e = n * [inf]
    sel_e = n * [-1]
    min_e[0] = 0

    mst = np.zeros_like(routes)

    for i in range(n):
        v = -1
        for j in range(n):
            if not used[j] and (v == -1 or min_e[j] < min_e[v]):
                v = j

        if min_e[v] == inf:
            raise ValueError('No MST')

        used[v] = True
        if sel_e[v] != -1:
            mst[v, sel_e[v]] = 1
            mst[sel_e[v], v] = 1

        for to in range(n):
            if routes[v][to] < min_e[to]:
                min_e[to] = routes[v][to]
                sel_e[to] = v

    return mst


class TsmProblemSolver:
    def __init__(self):
        self.__implemented_methods__ = {
            'brute_force': self.__brute_force_solve__,
            'greedy': self.__greedy_solve__,
            'christofides': self.__christofides_solve__
        }

    @staticmethod
    def __brute_force_solve__(routes: np.array) -> Tuple[float, np.array]:
        """
        Brute force implementation of TSM problem solution
        :param routes: numpy matrix of shape (X, X) where X is the total number of nodes
        :return: Tuple of total route distance and numpy array of shape (X) - the resulting node order
        """
        min_distance = 1e64
        min_route = None

        for p in permutations(range(routes.shape[0])):
            distance = 0
            for i, j in zip(p, p[1:]):
                distance += routes[i][j]
            distance += routes[p[0]][p[-1]]

            if distance < min_distance:
                min_distance = distance
                min_route = p

        min_route = list(min_route)
        min_route.append(min_route[0])

        return min_distance, np.array(min_route)

    @staticmethod
    def __greedy_solve__(routes: np.array) -> Tuple[float, np.array]:
        """
        Greedy solution for TSM problem

        :param routes: numpy matrix of shape (X, X) where X is the total number of nodes
        :return: Tuple of total route distance and numpy array of shape (X) - the resulting node order
        """
        node_number = routes.shape[0]
        start = randint(0, node_number - 1)
        used = np.zeros([node_number])
        used[start] = 1
        current_node = start
        min_distance = 0
        min_route = np.zeros([node_number + 1])
        min_route[0] = start

        i = 1
        while i < node_number:
            min_j = None
            min_road = 1e64

            for j, x in enumerate(routes[current_node]):
                if used[j] == 1 or j == current_node:
                    continue

                if x < min_road:
                    min_road = x
                    min_j = j

            used[min_j] = 1
            min_distance += min_road
            current_node = min_j
            min_route[i] = min_j
            i += 1

        min_route[-1] = start
        min_distance += routes[current_node][start]

        return min_distance, min_route

    @staticmethod
    def __christofides_solve__(routes: np.array) -> Tuple[float, np.ndarray]:
        """
        Christofides algorithm implementation for TSM problem solution

        :param routes: numpy matrix of shape (X, X) where X is the total number of nodes
        :return: Tuple of total route distance and numpy array of shape (X) - the resulting node order
        """
        mst = prims_mst(routes) * 2
        euler_cycle = find_euler_cycle(mst)
        # find_euler_cycle_rec(mst, 0)
        # hamiltonian_cycle = np.unique(np.array(GLOBAL_SHIT))
        hamiltonian_cycle = np.zeros((routes.shape[0] + 1,))
        used = [False] * len(routes)
        j = 0
        for i in range(len(euler_cycle)):
            if not used[euler_cycle[i]]:
                used[euler_cycle[i]] = True
                hamiltonian_cycle[j] = euler_cycle[i]
                j += 1

        min_distance = 0

        hamiltonian_cycle = hamiltonian_cycle.astype('int32')
        hamiltonian_cycle[-1] = hamiltonian_cycle[0]
        for i in range(1, len(hamiltonian_cycle)):
            min_distance += routes[hamiltonian_cycle[i]][hamiltonian_cycle[i - 1]]

        return min_distance, hamiltonian_cycle

    def solve(self, routes: np.array, method: str) -> Tuple[float, np.ndarray]:
        """
        Solve the TSM problem for euclidean graphs.

        :param routes: numpy matrix of shape (X, X) where X is the total number of nodes
        :param method: which method to use. Available methods: brute_force, greedy, christofides
        :return: Tuple of total route distance and numpy array of shape (X) - the resulting node order
        """
        if method not in self.__implemented_methods__:
            raise KeyError(f'Unrecognized method. Available methods: ', *self.__implemented_methods__.keys())
        return self.__implemented_methods__[method](routes)
