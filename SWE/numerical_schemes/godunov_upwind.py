""" This is an implementation of the Godunov upwind scheme, where we use 2 different Riemann solvers to compute the fluxes. 
1. The exact Riemann solver
2. The HLLC Riemann solver
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../SWE')
from aux_functions import file_manipulation, discritization, plotter, sampler

# Get a single sample returning the time step and result of a single time-step
def single_sample(solver, tolerance, iterations, dx, cfl, g, W, U):
    (dt, fluxes) = discritization.flux_at_boundaries(W, g, 3, solver, dx, tolerance, iteration, cfl)
    discritization.evolve(U, fluxes, dx, dt, 3)
    return (dt, U[1])

# Applies the numerical schemes to the entire domain from t=0 to t=t_end
def entire_domain(out_name, out_dir, bool_plot, x_len, break_pos, g, cells, riemann_int, riemann_str, tolerance, iterations, t_end, W_l, W_r):
    #discritization of the domain
    dx = x_len/cells
    (U,W) = discritization.discretize_initial_values(dx, cells, break_pos, W_l, W_r)
    CFL = 0.9
    t = 0
    end = False
    while t < t_end and not(end):
        #calculate the time step
        (delta_t, fluxes) = discritization.flux_at_boundaries(W, g, cells, riemann_int, dx, tolerance, iterations, CFL)
        if (delta_t + t > t_end):
            delta_t = t_end - t
            t = t + delta_t
            end = True
        else: 
            t = t + delta_t
        discritization.evolve(U, fluxes, dx, delta_t, cells) # using (8.8) page 143 Toro
        discritization.W_from_U(U, W)
    if (bool_plot):
        # calculate the exact solution
        exact_data = sampler.sample_exact(break_pos, x_len, t_end, cells, g, W_l, W_r, tolerance, iterations)
        plotter.plot(out_name, out_dir, x_len, t_end, cells, (True, False), exact_data, 0, W, riemann_str)
    return (U, W)

def main(terminal_arguments):
    # reading in terminal arguments
    try:
        read_file = open('input/' + terminal_arguments[1])
    except:
        print('Could not find input file, please remeber to execute the program from the root folder and specify the input file as first argument')
        sys.exit(1)
    (x_len, break_pos, g, cells, tolerance, iterations, t_end, h_l, u_l, psi_l, h_r, u_r, psi_r) = file_manipulation.extract(read_file) 

    if not (len(terminal_arguments) == 3):
        print('Please specify the Riemann solver to use as second argument, and provide no more than 2 arguments (input file and solver)')
        sys.exit(1)
    else:
        riemann_str = terminal_arguments[2]
        riemann_int = 0 # 0 = exact, 1 = HLLC
        if (riemann_str == 'exact'):
            pass
        elif (riemann_str == 'hllc'):
            riemann_int = 1
        else:
            print('Please specify \'exact\' or \'hllc\' as second argument. To choose the used riemann solver')
            sys.exit(1)
    
    (_,_) = entire_domain(os.path.splitext(terminal_arguments[1])[0], "output/godunov_upwind_results/" + riemann_str, True, x_len, break_pos, g, cells, riemann_int, riemann_str, tolerance, iterations, t_end, np.array([h_l, u_l, psi_l]), np.array([h_r, u_r, psi_r]))

if __name__ == '__main__':
    main(sys.argv)
    print("Completed successfully")