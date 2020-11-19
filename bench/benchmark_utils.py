import numpy as np


def generate_matrix(n: int) -> np.ndarray:
    x = np.linspace(-10.0, 10.0, 100 * n)
    y = np.linspace(-10.0, 10.0, 100 * n)

    np.random.shuffle(x)
    np.random.shuffle(y)

    x = x[:n]
    y = y[:n]

    space = np.array([x, y])
    space = space.transpose((1, 0))

    matrix = np.zeros([n, n])

    for i in range(n):
        for j in range(n):
            if i != j:
                matrix[i][j] = np.linalg.norm(space[i] - space[j])

    return matrix
