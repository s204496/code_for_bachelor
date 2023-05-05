""" This is an implementation of the Lax-Friedrichs scheme. 
This is a centered method, that does not depend on the Riemann solutions
We calculate the fluxes by:
F_{i+1/2}^{LF} = 1/2 * (F(U_i^n) + F(U_{i+1}^n)) - 1/2 * delta_x / delta_t * (U_{i+1} - U_i)
"""

import sys
import os
sys.path.append('../SWE')
from aux_functions import file_manipulation, discritization, plotter, sampler
import matplotlib.pyplot as plt

def lax_friedrich(bool_store_data, out_file, out_name, out_dir, bool_plot, x_len, break_pos, g, cells, tolerance, iterations, t_end, h_l, u_l, psi_l, h_r, u_r, psi_r):
    #discritization of the domain
    (U,W) = discritization.discritize_initial_values(x_len, cells, break_pos, h_l, u_l, psi_l, h_r, u_r, psi_r)
    CFL = 0.9
    t = 0
    end = False
    while t < t_end and not(end):
        #calculate the time step
        delta_x = x_len / cells
        delta_t = discritization.center_dt_wet(W, cells, g, CFL, delta_x)
        if (delta_t + t > t_end):
            delta_t = t_end - t
            end = True
        else: 
            t = t + delta_t
        fluxes = discritization.flux_lax_friedrich(W, U, g, cells, delta_x, delta_t)
        discritization.evolve(U, fluxes, x_len, delta_t, cells) # using (8.8) page 143 Toro
        discritization.W_from_U(U, W, cells)
    if (bool_plot):
        # calculate the exact solution
        exact_data = sampler.sample_exact(bool_store_data, out_file, break_pos, x_len, t_end, cells, g, h_l, u_l, psi_l, h_r, u_r, psi_r, tolerance, iterations)
        plotter.plot(out_name, out_dir, False, True, x_len, t_end, cells, (True, False), exact_data, 2, W, "not used")
    return (U, W)

def main(terminal_arguments):
    # reading in terminal arguments
    try:
        read_file = open('input/' + terminal_arguments[1])
    except:
        print('Could not find input file, please remeber to execute the program from the root folder and specify the input file as first argument')
        sys.exit(1)
    (x_len, break_pos, g, cells, tolerance, iterations, t_end, h_l, u_l, psi_l, h_r, u_r, psi_r) = file_manipulation.extract(read_file) 

    #open the file writen to
    try:
        #this creates the file if it does not exist, and overwrites it if it does
        out_file = open('output/lax_friedrich_results/' + terminal_arguments[2], 'w')
    except:
        print('Could not find output file, please specify the output file as second argument')
        sys.exit(1)

    (_,_) = lax_friedrich(False, out_file, os.path.splitext(terminal_arguments[2])[0], "output/lax_friedrich_results", True, x_len, break_pos, g, cells, tolerance, iterations, t_end, h_l, u_l, psi_l, h_r, u_r, psi_r)

if __name__ == '__main__':
    main(sys.argv)
    print("Completed successfully")