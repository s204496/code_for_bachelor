""" This is an implementation of the Lax-Friedrichs scheme. 
This is a centered method, that does not depend on the Riemann solutions
We calculate the fluxes by using the following formula:
F_{i+1/2}^{LF} = 1/2 * (F(U_i^n) + F(U_{i+1}^n)) - 1/2 * delta_x / delta_t * (U_{i+1} - U_i)
"""

import sys
import os
import numpy as np
sys.path.append('../SWE')
from aux_functions import file_manipulation, discritization, plotter, sampler, f
import matplotlib.pyplot as plt
from data_driven.aux_function import general_aux

# Applies the numerical schemes to the entire domain from t=0 to t=t_end
def entire_domain(out_name, out_dir, bool_plot, x_len, break_pos, g, cells, tolerance, iterations, t_end, W_l, W_r):
    model = None
    #discritization of the domain
    dx = x_len/cells
    (U,W) = discritization.discretize_initial_values(dx, cells, break_pos, W_l, W_r)
    CFL = 0.9
    t = 0
    end = False
    while t < t_end and not(end):
        #calculate the time step
        dt = discritization.center_dt(W, dx, g, CFL)
        if (dt + t > t_end):
            dt = t_end - t
            end = True
        else: 
            t = t + dt
        lax_flux = discritization.flux_lax(W, U, dx, cells, g, dt)
        discritization.evolve(U, lax_flux, dx, dt, cells) # using (8.8) page 143 Toro
        discritization.W_from_U(U, W)
    if (bool_plot):
        # calculate the exact solution
        exact_data = sampler.sample_exact(break_pos, x_len, t_end, cells, g, W_l, W_r, tolerance, iterations)
        plotter.plot(out_name, out_dir, x_len, t_end, cells, (True, False), exact_data, 1, W, "")
    return (U, W)

def main(terminal_arguments):
    # reading in terminal arguments
    try:
        read_file = open('input/' + terminal_arguments[1])
    except:
        print('Could not find input file, please remeber to execute the program from the root folder and specify the input file as first argument')
        sys.exit(1)
    (x_len, break_pos, g, cells, tolerance, iterations, t_end, h_l, u_l, psi_l, h_r, u_r, psi_r) = file_manipulation.extract(read_file) 
    save_dir = "output/numerical/lax_friedrich_results/"
    (_,_) = entire_domain(os.path.splitext(terminal_arguments[1])[0], save_dir, True, x_len, break_pos, g, cells, tolerance, iterations, t_end, np.array([h_l, u_l, psi_l]), np.array([h_r, u_r, psi_r]))

if __name__ == '__main__':
    main(sys.argv)
    print("Completed successfully - Lax Friedrich")