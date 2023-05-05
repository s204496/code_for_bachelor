""" This is an implementation of the TVD-WAF (Total Variation Diminishing - Weighted Average Flux), where we use 2 different Riemann solvers to compute the fluxes. 
1. The exact Riemann solver
2. The HLLC Riemann solver
"""

import sys
import os
sys.path.append('../SWE')
from aux_functions import file_manipulation, discritization, plotter, sampler
import matplotlib.pyplot as plt

def Waf(bool_store_data, out_file, out_name, out_dir, bool_plot, x_len, break_pos, g, cells, riemann_int, riemann_str, tolerance, iterations, t_end, h_l, u_l, psi_l, h_r, u_r, psi_r):
    #discritization of the domain
    (U,W) = discritization.discritize_initial_values(x_len, cells, break_pos, h_l, u_l, psi_l, h_r, u_r, psi_r)
    CFL = 0.9
    t = 0
    end = False
    while t < t_end and not(end):
        #calculate the time step
        (delta_t, boundary_flux) = discritization.boundary_fluxes(bool_store_data, out_file, W, g, cells, riemann_int, x_len, tolerance, iterations, CFL)
        if (delta_t + t > t_end):
            delta_t = t_end - t
            end = True
        else: 
            t = t + delta_t
        c_k = Riemann.wave_weights(W, g, riemann_int, cells)
        discritization.evolve(U, fluxes, x_len, delta_t, cells) # using (8.8) page 143 Toro
        discritization.W_from_U(U, W, cells)
    if (bool_plot):
        # calculate the exact solution
        exact_data = sampler.sample_exact(bool_store_data, out_file, break_pos, x_len, t_end, cells, g, h_l, u_l, psi_l, h_r, u_r, psi_r, tolerance, iterations)
        plotter.plot(out_name, out_dir, False, True, x_len, t_end, cells, (True, False), exact_data, 1, W, riemann_str)
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
    
    (_,_) = Waf(False, out_file, os.path.splitext(terminal_arguments[2])[0], "output/godunov_upwind_results", True, x_len, break_pos, g, cells, riemann_int, riemann_str, tolerance, iterations, t_end, h_l, u_l, psi_l, h_r, u_r, psi_r)

if __name__ == '__main__':
    main(sys.argv)
    print("Completed successfully")