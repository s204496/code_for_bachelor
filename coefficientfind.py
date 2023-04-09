from scipy.sparse.linalg import spsolve
import numpy as np

def fdcoeffV(k, xbar, x):
    n = len(x)
    A = np.ones((n, n))
    xrow = (x - xbar).reshape(-1, 1) # displacements as a column vector
    for i in range(1, n):
        A[i, :] = xrow[:, 0]**i / np.math.factorial(i)

    b = np.zeros((n, 1))
    b[k] = 1 # k'th derivative term remains
    c = np.linalg.solve(A, b) # solve system for coefficients
    c = c.flatten() # row vector

    return c

