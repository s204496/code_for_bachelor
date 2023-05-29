import sys
sys.path.append('../SWE')
import numpy as np
import pandas as pd
import random
from aux_functions import discritization 

def main(argv):
    try:
        samples = int(argv[1])
    except:
        print('Please specify the number of samples to generate as first argument')
        sys.exit(1)
    print('Generating ' + str(samples) + ' samples for the Lax-Friedirch lux ...')

    # Create an empty DataFrame
    df = pd.DataFrame(columns=['h_l', 'u_l', 'psi_l', 'delta_x', 'h_r', 'u_l', 'psi_l', 'delta_t', 'flux_h', 'flux_u', 'flux_psi'])
    # hardcoded values
    g = 9.8
    iterations = 50
    tolerance = 1e-10
    sizes = np.array([i*50 for i in range(2, 11)])
    sizes = np.append(sizes, np.array([i*100 for i in range(6, 11)]))
    sizes = np.append(sizes, np.array([i*200 for i in range(6, 21)]))
    sizes = np.append(sizes, np.array([i*500 for i in range(9, 16)]))
    x_len = 50
    cfl = 0.9

    i = 0
    while i < samples:
        cells = np.random.choice(sizes)
        dx = x_len/cells
        if (i % 25 == 0):
            psi = np.random.randint(2, size=1).astype(float)
            h_and_u = np.random.random(size=2)
            sign_u = np.random.choice([-1, 1], size=1)
            W = np.array([[h_and_u[0]*3, sign_u[0]*h_and_u[1]*6, psi.item()], [h_and_u[0]*3, sign_u[0]*h_and_u[1]*6, psi.item()]])
        else:
            psi_l, psi_r = np.random.randint(2, size=2).astype(float)
            h_and_u = np.random.random(size=4)
            sign_u = np.random.choice([-1, 1], size=2)
            W = np.array([[h_and_u[0]*3, sign_u[0]*h_and_u[1]*6, psi_l], [h_and_u[2]*3, sign_u[0]*h_and_u[3]*6, psi_r]])
        U = np.empty((2,3))
        discritization.U_from_W(U, W)
        dt = discritization.center_dt(W, dx, 0, g, cfl)
        flux = discritization.flux_lax_friedrich(W, U, dx, 0, g, dt)
        df.loc[i] = np.array([W[0][0], W[0][1], W[0][2], dx, W[1][0], W[1][1], W[1][2], dt, flux[0][0], flux[0][1], flux[0][2]])
        if (i % 10000 == 0):
            print("procent done: " + str(i/samples*100) + "%")
        i = i + 1 

    # Save dataframe to csv
    df.to_csv('data_driven/generated_data/lax_friedrich_flux.csv', index=False)

if __name__ == '__main__':
    main(sys.argv)
    print("Data Generation, Flux Lax Friedrich - Completed successfully")

