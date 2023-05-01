import numpy as np

def norm_2_FVM(exact, num, cells): 
    h = 1/cells
    e_v_local = (exact - num)**2
    e_global = np.sqrt(np.sum(e_v_local))
    # the h factor is to account for the fact that, the length of the error vectors grows with the number of cells, (see Leveque - Finite Difference - Appendix A.5)
    return h*e_global 