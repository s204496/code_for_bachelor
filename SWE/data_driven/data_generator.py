import sys
import numpy as np
import pandas as pd
import random
from numerical_schemes import tvd_waf, lax_friedrich, godunov_upwind

def dt_and_outdata(scheme, solver, delta_x, CFL, g, tolerance, iterations, W_l, W_c, W_r):
    match scheme:
        case 0:
            return lax_friedrich.single_sample(delta_x, CFL, g, W_l, W_c, W_r)
        case 1:
            return godunov_upwind.single_sample(solver, tolerance, iterations, delta_x, CFL, g, W_l, W_c, W_r)
        case 2:
            return tvd_waf.single_sample(solver, tolerance, iterations, delta_x, CFL, g, W_l, W_c, W_r)
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

    for i in range(samples):
        h_and_u = np.random.random(size=6)
        h_l, h_c, h_r, u_l, u_c, u_r = h_and_u[0]*10, h_and_u[1]*10, h_and_u[2]*10, h_and_u[3]*10, h_and_u[4]*10, h_and_u[5]*10
        psi_l, psi_c, psi_r = np.random.randint(2, size=3).astype(float)
        W_l, W_c, W_r = np.array([h_l, u_l, psi_l]), np.array([h_c, u_c, psi_c]), np.array([h_r, u_r, psi_r])
        hu_l, hu_c, hu_r = h_l*u_l, h_c*u_c, h_r*u_r
        hpsi_l, hpsi_c, hpsi_r = h_l*psi_l, h_c*psi_c, h_r*psi_r
        U_l, U_c, U_r = np.array([h_l, hu_l, hpsi_l]), np.array([h_c, hu_c, hpsi_c]), np.array([h_r, hu_r, hpsi_r])
        delta_x = x_len/np.random.choice(sizes)
        CFL = 0.9

        delta_t, h_out, hu_out, hpsi_out = dt_and_outdata(scheme, solver, delta_x, CFL, g, W_l, W_c, W_r) 
        # Generate data and add rows to the DataFrame
        df.loc[i] = np.array([h_l, h_c, h_r, u_l, u_c, u_r, psi_l, psi_c, psi_r, hu_l, hu_c, hu_r, hpsi_l, hpsi_c, hpsi_r, delta_x, delta_t, CFL, h_out, hu_out, hpsi_out])

    # Display the resulting DataFrame
    print(df)

if __name__ == '__main__':
    main(sys.argv)
    print("Data Generation - Completed successfully")

