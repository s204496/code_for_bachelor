import sys
import numpy as np
import pandas as pd
import random
from numerical_schemes import tvd_waf, lax_friedrich, godunov_upwind

# Get a single sample returning the time step and result of a single cell delta_t later.
def dt_and_outdata(scheme, solver, tolerance, iterations, dx, cfl, g, W, U):
    match scheme:
        case 0:
            return lax_friedrich.single_sample(dx, cfl, g, W, U)
        case 1:
            return godunov_upwind.single_sample(solver, tolerance, iterations, dx, CFL, g, W, U)
        case 2:
            return tvd_waf.single_sample(solver, tolerance, iterations, dx, CFL, g, W_l, W_c, W_r)
    print("Invalid scheme, failed in dt_and_outdata")
    sys.exit(1)

def main(argv):

    try:
        samples = int(argv[1])
    except:
        print('Please specify the number of samples to generate as first argument')
        sys.exit(1)
    try:
        scheme = int(argv[2]) # 0 = Godunov, 1=Lax-friedrich, 2 = TVD WAF
    except:
        print('Please specify the scheme to generate data for as second argument\n0=Lax-friedrich, 1 = Godunov, 2 = TVD WAF')
    if (scheme == 1 or scheme == 2):
        try:
            solver = int(argv[3]) # 0 = exact, 1 = HLLC 
        except:
            print("Please specify the Riemann solver to use as third argument\n0 = exact, 1 = HLLC")
    else:
        solver = None 

    if not(scheme == 0):
        print('Generating ' + str(samples) + ' samples for scheme ' + str(scheme) + ' and solver ' + str(solver) + ' ...')
    else: 
        print('Generating ' + str(samples) + ' samples for scheme ' + str(scheme) + ' ...')

    # Create an empty DataFrame
    if (scheme == 2):
        df = pd.DataFrame(columns=['h_ll, h_l', 'h_c', 'h_r', 'h_rr', 'u_ll', 'u_l', 'u_c', 'u_r', 'u_rr', 
                               'psi_ll', 'psi_l', 'psi_c', 'psi_r','spi_rr', 'hu_ll' 'hu_l', 'hu_c', 'hu_r', 'hu_rr', 
                               'hpsi_ll', 'hpsi_l', 'hpsi_c', 'hpsi_r', 'hspi_rr',
                               'delta_x', 'delta_t', 'CFL', 'h_out', 'hu_out', 'hpsi_out'])
    else:
        df = pd.DataFrame(columns=['h_l', 'h_c', 'h_r', 'u_l', 'u_c', 'u_r', 'psi_l', 'psi_c', 'psi_r',
                               'hu_l', 'hu_c', 'hu_r', 'hpsi_l', 'hpsi_c', 'hpsi_r',
                               'delta_x', 'delta_t', 'CFL', 'h_out', 'hu_out', 'hpsi_out'])

    # These are to varying the delta_x
    sizes = np.array([i*50 for i in range(2, 11)])
    sizes = np.append(sizes, np.array([i*100 for i in range(6, 11)]))
    sizes = np.append(sizes, np.array([i*200 for i in range(6, 21)]))
    sizes = np.append(sizes, np.array([i*500 for i in range(9, 16)]))
    # hardcoded values
    x_len = 50
    g = 9.8
    iterations = 50
    tolerance = 1e-7
    cfl = 0.9

    for i in range(samples):
        cells = np.random.choice(sizes)
        dx = x_len/cells
        psi_split = 1+np.random.randint(2, size=1).astype(float)
        psi_l, psi_r = np.random.randint(2, size=2).astype(float)
        if (scheme == 2):
            h_and_u = np.random.random(size=10)
            psi = np.empty(5, dtype=float)
            psi[0:psi_split+1], psi[psi_split+1:] = psi_l, psi_r 
            W_pre = np.array([[h_and_u[0]*10, h_and_u[1]*10, psi[0]], [h_and_u[2]*10, h_and_u[3]*10, psi[1]], [h_and_u[4]*10, h_and_u[5]*10, psi[2]], [h_and_u[6]*10, h_and_u[7]*10, psi[3]], [h_and_u[8]*10, h_and_u[9]*10, psi[4]]])
            W = np.empty((7,3), dtype=float)
            W[1:-1] = W_pre
            W[0], W[-1] = W_pre[0], W_pre[-1]
        else:
            h_and_u = np.random.random(size=6)
            psi = np.empty(3, dtype=float)
            psi[0:psi_split], psi[psi_split:] = psi_l, psi_r 
            W = np.array([[h_and_u[0]*10, h_and_u[1]*10, psi[0]], [h_and_u[2]*10, h_and_u[3]*10, psi[1]], [h_and_u[4]*10, h_and_u[5]*10, psi[2]]])    
        U = W[:,0], W[:,0]*W[:,1], W[:,0]*W[:,2]
        (delta_t, U_out) = dt_and_outdata(scheme, solver, tolerance, iterations, dx, cfl, g, W, U) 

        if (scheme == 2):
            df.loc[i] = np.array([h_l, h_c, h_r, u_l, u_c, u_r, psi_l, psi_c, psi_r, hu_l, hu_c, hu_r, hpsi_l, hpsi_c, hpsi_r, delta_x, delta_t, cfl, h_out, hu_out, hpsi_out])
        else (scheme == 1):
            df.loc[i] = np.array([h_l, h_c, h_r, u_l, u_c, u_r, psi_l, psi_c, psi_r, hu_l, hu_c, hu_r, hpsi_l, hpsi_c, hpsi_r, delta_x, delta_t, cfl, h_out, hu_out, hpsi_out])

    # Display the resulting DataFrame
    print(df)

if __name__ == '__main__':
    main(sys.argv)
    print("Data Generation - Completed successfully")

