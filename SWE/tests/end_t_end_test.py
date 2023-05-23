import sys
import numpy as np
sys.path.append('SWE')
from numerical_schemes import lax_friedrich, godunov_upwind, tvd_waf

def test_allways_true() -> None:
    assert True

def test_end_to_end_lax() -> None:
    W = np.array([[0.5,1,1], [0.5,1,1],[10,-2,0], [3,2,0], [3,2,0]])
    U = np.empty((5,3))
    U[:,0], U[:,1], U[:,2] = W[:,0], W[:,0]*W[:,1], W[:,0]*W[:,2] 
    (dt, U) = lax_friedrich.single_sample(0.1, 0.9, 9.8, W, U) 
    assert np.all(np.isclose(U, [1.54200798, 1.19371525, 0.26890837])) and np.isclose(dt, 0.0075633462159047865)

test_end_to_end_lax()