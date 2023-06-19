# This script generates data, for a NN that take in h_l, u_l, h_r, u_r, and outputs Fi-1/2 for the boundary between them with 2.48 as emulation.

import sys
sys.path.append('../SWE')
import numpy as np
import pandas as pd
from aux_functions import exact_riemann_solver, hllc_riemann

def main(argv):
    solver = 0
    try:
        sample_choice = int(argv[1])
        if (sample_choice == 0):
            samples = 25000
            samples_str = '25k'
        elif (sample_choice == 1):
            samples = 200000
            samples_str = '200k'
        elif (sample_choice == 2):
            samples = 1600000
            samples_str = '1d6M'
        else:
            print('Please specify the number of samples to generate as first argument\n0: for 25.000 samples\n1: for 200.000 samples\n2: for 1.600.000 samples')
            sys.exit(1)
    except:
        print('Please specify the number of samples to generate as first argument\n0: for 25.000 samples\n1: for 200.000 samples\n2: for 1.600.000 samples')
        sys.exit(1)
    try:
        if (argv[2] == 'exact'):
            pass
        elif (argv[2] == 'hllc'):
            solver = 1
        else:
            print('Please specify the \'exact\' or \'hllc\' as second argument')
            sys.exit(1)
    except:
        print('Please specify the \'exact\' or \'hllc\' as second argument')
        sys.exit(1)
    print('Generating ' + str(samples) + ' samples for the ' + argv[2] + ' Riemann solver')
    # Create an empty DataFrame
    df = pd.DataFrame(columns=['h_l', 'u_l', 'h_r', 'u_r', 'flux_h', 'flux_hu'])
    g = 9.8
    iterations = 50
    tolerance = 1e-10

    i = 0
    while i < samples:
        h_and_u = np.random.random(size=4)
        sign_u = np.random.choice([-1, 1], size=2)
        W_pre = np.array([[h_and_u[0]*3, sign_u[0]*h_and_u[1]*6, 0.0], [h_and_u[2]*3, sign_u[1]*h_and_u[3]*6, 0.0]])
        if (solver == 0):
            (_, (_, _, _), boundary_flux, _) = exact_riemann_solver.solve(0.0, W_pre[0], W_pre[1], g, tolerance, iterations)
        else:
            (_, (_, _, _), boundary_flux, (_ , _)) = hllc_riemann.solve(W_pre[0], W_pre[1], g)
        df.loc[i] = np.array([W_pre[0][0], W_pre[0][1], W_pre[1][0], W_pre[1][1], boundary_flux[0], boundary_flux[1]])
        if (i % 10000 == 0):
            print("procent done: " + str(i/samples*100) + "%")
        i = i + 1 

    # Save dataframe to csv
    df.to_csv('data_driven/generated_data/godunov_flux_' + argv[2] + '_' + samples_str  + '.csv' , index=False)

if __name__ == '__main__':
    main(sys.argv)
    print("Data Generation - Completed successfully")
