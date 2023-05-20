""" This is an implementation of the TVD-WAF (Total Variation Diminishing - Weighted Average Flux), where we use 2 different Riemann solvers to compute the fluxes. 
1. The exact Riemann solver
2. The HLLC Riemann solver
"""

import sys
import os
sys.path.append('../SWE')
from aux_functions import file_manipulation, discritization, plotter, sampler
import numpy as np 
import matplotlib.pyplot as plt

def TVD_WAF_entire_domain(out_name, out_dir, bool_plot, x_len, break_pos, g, cells, riemann_int, riemann_str, tolerance, iterations, t_end, W_l, W_r):
    #discritization of the domain
    (U,W) = discritization.discretize_initial_values(x_len, cells, break_pos, W_l, W_r)
    CFL = 0.9
    t = 0
    end = False
    while t < t_end and not(end):
        #calculate the time step
        (delta_t, boundary_flux) = discritization.flux_at_boundaries(W, g, cells, riemann_int, x_len, tolerance, iterations, CFL)
        if (delta_t + t > t_end):
            delta_t = t_end - t
            t = t + delta_t 
            end = True
        else: 
            t = t + delta_t
        waf_fluxes = discritization.flux_WAF_TVD(W, g, riemann_int, cells, delta_t, x_len/cells, boundary_flux, tolerance, iterations)
        discritization.evolve(U, waf_fluxes, x_len, delta_t, cells) # using (8.8) page 143 Toro
        discritization.W_from_U(U, W)
    if (bool_plot):
        # calculate the exact solution
        exact_data = sampler.sample_exact(break_pos, x_len, t_end, cells, g, h_l, u_l, psi_l, h_r, u_r, psi_r, tolerance, iterations)
        plotter.plot(out_name, out_dir, x_len, t_end, cells, (True, False), np.array(exact_data), 2, np.array(W), riemann_str)
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
        out_file = open('output/WAF_results/' + terminal_arguments[2], 'w')
    except:
        print('Could not find output file, please specify the output file as second argument')
        sys.exit(1)
    if not (len(terminal_arguments) == 4):
        print('Please specify the Riemann solver to use as third argument, and provide no more than 3 arguments (input file, output file and solver)')
        sys.exit(1)
    else:
        riemann_str = terminal_arguments[3]
        riemann_int = 0 # 0 = exact, 1 = HLLC
        if (riemann_str == 'exact'):
            pass
        elif (riemann_str == 'HLLC'):
            riemann_int = 1
        else:
            print('Please specify exact or HLLC as third argument. To choose the used riemann solver')
            sys.exit(1)
    
    (_,_) = TVD_WAF_entire_domain(False, out_file, os.path.splitext(terminal_arguments[2])[0], "output/WAF_results", True, x_len, break_pos, g, cells, riemann_int, riemann_str, tolerance, iterations, t_end, np.array([h_l, u_l, psi_l]), np.array([h_r, u_r, psi_r]))

if __name__ == '__main__':
    main(sys.argv)
    print("Completed successfully")