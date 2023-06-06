import sys
sys.path.append('../SWE')
import numpy as np
import pandas as pd
import random
from aux_functions import exact_riemann_solver, hllc_riemann, wet_bed

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
    print('Generating ' + str(samples) + ' samples for the ' + argv[2] + ' Riemann...')
    # Create an empty DataFrame
    df = pd.DataFrame(columns=['h_l', 'u_l', 'h_r', 'u_r', 'h_s'])
    g = 9.8
    iterations = 50
    tolerance = 1e-10

    i = 0
    while i < samples:
        #if (i % 25 == 0):
        #    psi = np.random.randint(2, size=1).astype(float)
        #    h_and_u = np.random.random(size=2)
        #    sign_u = np.random.choice([-1, 1], size=1)
        #    df.loc[i] = np.array([h_and_u[0]*3, sign_u[0]*h_and_u[1]*6, h_and_u[0]*3, sign_u[0]*h_and_u[1]*6, h_and_u[0]*3])
        #else:
        psi_l, psi_r = np.random.randint(2, size=2).astype(float)
        h_and_u = np.random.random(size=4)
        sign_u = np.random.choice([-1, 1], size=2)
        W_pre = np.array([[h_and_u[0]*3, sign_u[0]*h_and_u[1]*6, psi_l], [h_and_u[2]*3, sign_u[0]*h_and_u[3]*6, psi_r]])
        a_l = np.sqrt(g*W_pre[0][0].item())
        a_r = np.sqrt(g*W_pre[1][0].item())
        if (solver == 0):
            W_out = wet_bed.calculate(g, tolerance, iterations, W_pre[0][0].item(), W_pre[0][1].item(), a_l, W_pre[1][0].item(), W_pre[1][1].item(), a_r)
            h_s =  W_out[0]
        else:
            hllc_riemann_v = hllc_riemann.solve(W_pre[0], W_pre[1], 9.8)
            h_s = hllc_riemann_v[3][0]
        df.loc[i] = np.array([W_pre[0][0], W_pre[0][1], W_pre[1][0], W_pre[1][1], h_s])
        if (i % 10000 == 0):
            print("procent done: " + str(i/samples*100) + "%")
        i = i + 1 

    # Save dataframe to csv
    temp = 'exact' if solver == 0 else 'hllc'
    df.to_csv('data_driven/generated_data/riemann_' + temp + '.csv', index=False)

if __name__ == '__main__':
    main(sys.argv)
    print("Data Generation - Completed successfully")
