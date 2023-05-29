import sys
sys.path.append('../SWE')
import numpy as np
import pandas as pd
import random
from aux_functions import exact_riemann_solver, hllc_riemann

def main(argv):
    try:
        samples = int(argv[1])
    except:
        print('Please specify the number of samples to generate as first argument')
        sys.exit(1)
    try:
        if ('exact' == argv[2]):
            solver = 0 
        elif ('hllc' == argv[2]): 
            solver = 1
    except:
        print("Please specify the Riemann solver to use as second argument \'exact\' or \'hllc\'")
        sys.exit(1)
    print('Generating ' + str(samples) + ' samples for the ' + argv[2] + ' Riemann solver ...')

    # Create an empty DataFrame
    df = pd.DataFrame(columns=['h_l', 'u_l', 'psi_l', 'h_r', 'u_l', 'psi_l', 'h_out', 'u_out', 'psi_out'])
    g = 9.8
    iterations = 50
    tolerance = 1e-10

    i = 0
    while i < samples:
        if (i % 25 == 0):
            psi = np.random.randint(2, size=1).astype(float)
            h_and_u = np.random.random(size=2)
            sign_u = np.random.choice([-1, 1], size=1)
            df.loc[i] = np.array([h_and_u[0]*3, sign_u[0]*h_and_u[1]*6, psi.item(), h_and_u[0]*3, sign_u[0]*h_and_u[1]*6, psi.item(), h_and_u[0]*3, sign_u[0]*h_and_u[1]*6, psi.item()])
        else:
            psi_l, psi_r = np.random.randint(2, size=2).astype(float)
            h_and_u = np.random.random(size=4)
            sign_u = np.random.choice([-1, 1], size=2)
            W_pre = np.array([[h_and_u[0]*3, sign_u[0]*h_and_u[1]*6, psi_l], [h_and_u[2]*3, sign_u[0]*h_and_u[3]*6, psi_r]])
            W_out = np.zeros(3)
            if (solver == 0):
                output_data = exact_riemann_solver.solve(0.0, W_pre[0], W_pre[1], g, tolerance, iterations)
                W_out = output_data[1]
                if(output_data[0]):
                    continue
            else:
                output_data = hllc_riemann.solve(W_pre[0], W_pre[1], g)
                W_out = output_data[1]
                if(output_data[0]):
                    continue
            df.loc[i] = np.array([W_pre[0][0], W_pre[0][1], W_pre[0][2], W_pre[1][0], W_pre[1][1], W_pre[1][2], W_out[0], W_out[1], W_out[2]])
        if (i % 10000 == 0):
            print("procent done: " + str(i/samples*100) + "%")
        i = i + 1 

    # Save dataframe to csv
    temp = 'exact' if solver == 0 else 'hllc'
    df.to_csv('data_driven/generated_data/riemann_' + temp + '.csv', index=False)

if __name__ == '__main__':
    main(sys.argv)
    print("Data Generation - Completed successfully")

