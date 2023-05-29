import sys
import numpy as np
sys.path.append('SWE')
from numerical_schemes import lax_friedrich, godunov_upwind, tvd_waf

#### LAX-FRIEDRICH TESTS: END TO END ####

# Test1: Just a general test for Lax-Friedrichs
def test_end_to_end_lax1() -> None:
    W = np.array([[0.5,1,1], [0.5,1,1],[10,-2,0], [3,2,0], [3,2,0]])
    U = np.empty((5,3))
    U[:,0], U[:,1], U[:,2] = W[:,0], W[:,0]*W[:,1], W[:,0]*W[:,2] 
    (dt, U) = lax_friedrich.single_sample(0.1, 0.9, 9.8, W, U) 
    assert np.all(np.isclose(U, [1.54200798, 1.19371525, 0.26890837])) and np.isclose(dt, 0.0075633462159047865)

# Test2: Left sonic rarefaction test for Lax-Friedrichs
def test_end_to_end_lax2() -> None:
    W = np.array([[1.0,2.5,1], [1.0,2.5,1],[0.1,0.0,0], [0.1,0.0,0], [0.1,0.0,0]])
    U = np.empty((5,3))
    U[:,0], U[:,1], U[:,2] = W[:,0], W[:,0]*W[:,1], W[:,0]*W[:,2] 
    (dt, U) = lax_friedrich.single_sample(0.1, 0.9, 9.8, W, U) 
    assert np.all(np.isclose(U, [0.74980481, 2.13721326, 0.69980481])) and np.isclose(dt, 0.015984384553513663)

# Test3: Right sonic rarefaction test for Lax-Friedrichs
def test_end_to_end_lax3() -> None:
    W = np.array([[0.1,0.0,1], [0.1,0.0,1],[0.1,0.0,1], [1.0,-2.5,1], [1.0,-2.5,1]])
    U = np.empty((5,3))
    U[:,0], U[:,1], U[:,2] = W[:,0], W[:,0]*W[:,1], W[:,0]*W[:,2] 
    (dt, U) = lax_friedrich.single_sample(0.1, 0.9, 9.8, W, U) 
    assert np.all(np.isclose(U, [0.74980481, -2.13721326,  0.74980481])) and np.isclose(dt, 0.015984384553513663)

# Test4: 2 shock test for Lax-Friedrichs
def test_end_to_end_lax4() -> None:
    W = np.array([[3.0,0.6390096505,1], [3.0,0.6390096505,1],[3.0,0.6390096505,1], [3.0,-0.6390096505,1], [3.0,-0.6390096505,1]])
    U = np.empty((5,3))
    U[:,0], U[:,1], U[:,2] = W[:,0], W[:,0]*W[:,1], W[:,0]*W[:,2] 
    (dt, U) = lax_friedrich.single_sample(0.1, 0.9, 9.8, W, U) 
    assert np.all(np.isclose(U, [3.28465155e+00, 2.22044605e-16, 3.28465155e+00])) and np.isclose(dt, 0.014848578318319112)

# Test5: 2 rarefaction and nearly dry test for Lax-Friedrichs
def test_end_to_end_lax5() -> None:
    W = np.array([[1.0,-5,1], [1.0,-5,1],[1.0,-5,1], [1.0,5.0,0], [1.0,5.0,0]])
    U = np.empty((5,3))
    U[:,0], U[:,1], U[:,2] = W[:,0], W[:,0]*W[:,1], W[:,0]*W[:,2] 
    (dt, U) = lax_friedrich.single_sample(0.1, 0.9, 9.8, W, U) 
    assert np.all(np.isclose(U, [0.44652817, 0., 0.22326409])) and np.isclose(dt, 0.011069436502304375)

#### GODUNOV-UPWIND TESTS, exact solver: END TO END ####

# Test1: Just a general test for Godunov-Upwind
def test_end_to_end_godunov1_exact() -> None:
    W = np.array([[0.5,1,1], [0.5,1,1],[10,-2,0], [3,2,0], [3,2,0]])
    U = np.empty((5,3))
    U[:,0], U[:,1], U[:,2] = W[:,0], W[:,0]*W[:,1], W[:,0]*W[:,2] 
    (dt, U) = godunov_upwind.single_sample(0, 10e-8, 50, 0.1, 0.9, 9.8, W, U) 
    assert np.all(np.isclose(U, [5.50381147, -2.18216853, 0.])) and np.isclose(dt, 0.0075633462159047865)

# Test2: Left sonic rarefaction test for Godunov-Upwind
def test_end_to_end_godunov2_exact() -> None:
    W = np.array([[1.0,2.5,1], [1.0,2.5,1],[0.1,0.0,0], [0.1,0.0,0], [0.1,0.0,0]])
    U = np.empty((5,3))
    U[:,0], U[:,1], U[:,2] = W[:,0], W[:,0]*W[:,1], W[:,0]*W[:,2] 
    (dt, U) = godunov_upwind.single_sample(0, 10e-8, 50, 0.1, 0.9, 9.8, W, U) 
    assert np.all(np.isclose(U, [0.5062241, 1.77163035, 0.4062241])) and np.isclose(dt, 0.015984384553513663)

# Test3: Right sonic rarefaction test for Godunov-Upwind
def test_end_to_end_godunov3_exact() -> None:
    W = np.array([[0.1,0.0,1], [0.1,0.0,1],[0.1,0.0,1], [1.0,-2.5,1], [1.0,-2.5,1]])
    U = np.empty((5,3))
    U[:,0], U[:,1], U[:,2] = W[:,0], W[:,0]*W[:,1], W[:,0]*W[:,2] 
    (dt, U) = godunov_upwind.single_sample(0, 10e-8, 50, 0.1, 0.9, 9.8, W, U) 
    assert np.all(np.isclose(U, [0.5062241, -1.77163035, 0.5062241])) and np.isclose(dt, 0.015984384553513663)

# Test4: 2 shock test for Godunov-Upwind
def test_end_to_end_godunov4_exact() -> None:
    W = np.array([[3.0,0.6390096505,1], [3.0,0.6390096505,1],[3.0,0.6390096505,1], [3.0,-0.6390096505,1], [3.0,-0.6390096505,1]])
    U = np.empty((5,3))
    U[:,0], U[:,1], U[:,2] = W[:,0], W[:,0]*W[:,1], W[:,0]*W[:,2] 
    (dt, U) = godunov_upwind.single_sample(0, 10e-8, 50, 0.1, 0.9, 9.8, W, U) 
    assert np.all(np.isclose(U, [3.28465155, 0.41587866, 3.28465155])) and np.isclose(dt, 0.014848578318319112)

# Test5: 2 rarefaction and nearly dry test for Godunov-Upwind
def test_end_to_end_godunov5_exact() -> None:
    W = np.array([[1.0,-5,1], [1.0,-5,1],[1.0,-5,1], [1.0,5.0,0], [1.0,5.0,0]])
    U = np.empty((5,3))
    U[:,0], U[:,1], U[:,2] = W[:,0], W[:,0]*W[:,1], W[:,0]*W[:,2] 
    (dt, U) = godunov_upwind.single_sample(0, 10e-8, 50, 0.1, 0.9, 9.8, W, U) 
    assert np.all(np.isclose(U, [0.44652817, -1.69113096, 0.44652817])) and np.isclose(dt, 0.011069436502304375)

#### GODUNOV-UPWIND TESTS, hllc solver: END TO END ####

# Test1: Just a general test for Godunov-Upwind
def test_end_to_end_godunov1_hllc() -> None:
    W = np.array([[0.5,1,1], [0.5,1,1],[10,-2,0], [3,2,0], [3,2,0]])
    U = np.empty((5,3))
    U[:,0], U[:,1], U[:,2] = W[:,0], W[:,0]*W[:,1], W[:,0]*W[:,2] 
    (dt, U) = godunov_upwind.single_sample(1, 10e-8, 50, 0.1, 0.9, 9.8, W, U) 
    assert np.all(np.isclose(U, [4.66587718, 2.5190076, 0.])) and np.isclose(dt, 0.0075633462159047865)

# Test2: Left sonic rarefaction test for Godunov-Upwind
def test_end_to_end_godunov2_hllc() -> None:
    W = np.array([[1.0,2.5,1], [1.0,2.5,1],[0.1,0.0,0], [0.1,0.0,0], [0.1,0.0,0]])
    U = np.empty((5,3))
    U[:,0], U[:,1], U[:,2] = W[:,0], W[:,0]*W[:,1], W[:,0]*W[:,2] 
    (dt, U) = godunov_upwind.single_sample(1, 10e-8, 50, 0.1, 0.9, 9.8, W, U) 
    assert np.all(np.isclose(U, [0.5062241, 1.77163035, 0.4062241])) and np.isclose(dt, 0.015984384553513663)

# Test3: Right sonic rarefaction test for Godunov-Upwind
def test_end_to_end_godunov3_hllc() -> None:
    W = np.array([[0.1,0.0,1], [0.1,0.0,1],[0.1,0.0,1], [1.0,-2.5,1], [1.0,-2.5,1]])
    U = np.empty((5,3))
    U[:,0], U[:,1], U[:,2] = W[:,0], W[:,0]*W[:,1], W[:,0]*W[:,2] 
    (dt, U) = godunov_upwind.single_sample(1, 10e-8, 50, 0.1, 0.9, 9.8, W, U) 
    assert np.all(np.isclose(U, [0.5062241, -1.77163035, 0.5062241])) and np.isclose(dt, 0.015984384553513663)

# Test4: 2 shock test for Godunov-Upwind
def test_end_to_end_godunov4_hllc() -> None:
    W = np.array([[3.0,0.6390096505,1], [3.0,0.6390096505,1],[3.0,0.6390096505,1], [3.0,-0.6390096505,1], [3.0,-0.6390096505,1]])
    U = np.empty((5,3))
    U[:,0], U[:,1], U[:,2] = W[:,0], W[:,0]*W[:,1], W[:,0]*W[:,2] 
    (dt, U) = godunov_upwind.single_sample(1, 10e-8, 50, 0.1, 0.9, 9.8, W, U) 
    assert np.all(np.isclose(U, [3.28465155, 0.41570326, 3.28465155])) and np.isclose(dt, 0.014848578318319112)

# Test5: 2 rarefaction and nearly dry test for Godunov-Upwind
def test_end_to_end_godunov5_hllc() -> None:
    W = np.array([[1.0,-5,1], [1.0,-5,1],[1.0,-5,1], [1.0,5.0,0], [1.0,5.0,0]])
    U = np.empty((5,3))
    U[:,0], U[:,1], U[:,2] = W[:,0], W[:,0]*W[:,1], W[:,0]*W[:,2] 
    (dt, U) = godunov_upwind.single_sample(1, 10e-8, 50, 0.1, 0.9, 9.8, W, U) 
    assert np.all(np.isclose(U, [0.44652817, -0.5, 0.44652817])) and np.isclose(dt, 0.011069436502304375)

