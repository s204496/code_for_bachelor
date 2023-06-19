""" This is an implementation of the TVD-WAF (Total Variation Diminishing - Weighted Average Flux), where we use 2 different Riemann solvers to compute the fluxes. 
1. The exact Riemann solver
2. The HLLC Riemann solver
"""

import sys
import os
sys.path.append('../SWE')
from aux_functions import file_manipulation, discritization, plotter, sampler
from data_driven.aux_function import general_aux
import numpy as np 

# Applies the numerical schemes to the entire domain from t=0 to t=t_end
def entire_domain(out_name, out_dir, bool_plot, x_len, break_pos, g, cells, riemann_solver, riemann_str, tolerance, iterations, t_end, W_l, W_r):
    model = None
    if riemann_solver == 2:
        model = general_aux.load_model('data_driven/models/riemann_exact_FFNN_shallow_approach0.pt', 'cpu', 'ffnn_riemann') # CPU can be changed if one has a Nvidia GPU
    #discritization of the domain
    dx = x_len/cells
    (U,W) = discritization.discretize_initial_values(dx, cells, break_pos, W_l, W_r)
    CFL = 0.9
    t = 0
    end = False
    while t < t_end and not(end):
        #calculate the time step
        if riemann_solver == 2:
            (delta_t, fluxes) = discritization.flux_at_boundaries_data_driven(W, g, cells, dx, CFL, model)
        else:
            (delta_t, fluxes) = discritization.flux_at_boundaries(W, g, cells, riemann_solver, dx, tolerance, iterations, CFL)
        if (delta_t + t > t_end):
            delta_t = t_end - t
            t = t + delta_t 
            end = True
        else: 
            t = t + delta_t
        if riemann_solver == 2:
            waf_fluxes = discritization.flux_waf_tvd(W, g, 0, cells, delta_t, x_len/cells, fluxes, tolerance, iterations)
        else:
            waf_fluxes = discritization.flux_waf_tvd(W, g, riemann_solver, cells, delta_t, x_len/cells, fluxes, tolerance, iterations)
        discritization.evolve(U, waf_fluxes, dx, delta_t, cells) # using (8.8) page 143 Toro
        discritization.W_from_U(U, W)
    if (bool_plot):
        # calculate the exact solution
        exact_data = sampler.sample_exact(break_pos, x_len, t_end, cells, g, W_l, W_r, tolerance, iterations)
        if riemann_solver == 2:
            plotter.plot(out_name, out_dir, x_len, t_end, cells, (True, False), exact_data, 2, W, riemann_str, True)
        else:
            plotter.plot(out_name, out_dir, x_len, t_end, cells, (True, False), exact_data, 2, W, riemann_str, False)
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
        print('Please specify the Riemann solver to use as second argument, and provide no more than 2 arguments (input file, and solver)\nSecond argument:\n\'exact\'\n\'hllc\'\n\'data-driven\'')
        sys.exit(1)
    else:
        riemann_str = terminal_arguments[2]
        riemann_solver = 0 # 0 = exact, 1 = HLLC, 2 = data-driven
        if (riemann_str == 'exact'):
            pass
        elif (riemann_str == 'hllc'):
            riemann_solver = 1
        elif (riemann_str == 'data-driven'):
            riemann_solver = 2 
        else:
            print('Please specify exact or hllc as third argument. To choose the used riemann solver')
            sys.exit(1)
    save_dir = "output/numerical/tvd_waf_results/" + riemann_str
    if riemann_solver == 2:
        save_dir = "output/data_driven/riemann/tvd_waf_approach4_handle_dry/ffnn"
    
    (_,_) = entire_domain(os.path.splitext(terminal_arguments[1])[0], save_dir, True, x_len, break_pos, g, cells, riemann_solver, riemann_str, tolerance, iterations, t_end, np.array([h_l, u_l, psi_l]), np.array([h_r, u_r, psi_r]))

if __name__ == '__main__':
    main(sys.argv)
    print("Completed successfully")