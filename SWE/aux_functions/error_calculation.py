import numpy as np

# This calculates the 2 Norm, for errors based on Finite Difference methods for ODEs and PDEs - Leveque - Appendix 5.A
def norm_2_fvm(exact, num, cells, x_len): 
    h = x_len/cells
    e_v_local = (exact - num)**2
    e_global = np.sqrt(np.sum(e_v_local))
    return h*e_global 