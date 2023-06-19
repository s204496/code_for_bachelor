""" This is an implementation of the Godunov upwind scheme, where we use 2 different Riemann solvers to compute the fluxes. 
1. The exact Riemann solver
2. The HLLC Riemann solver
One can also choose between the numerical solver, or using
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../SWE')
from aux_functions import file_manipulation, discritization, plotter, sampler
from data_driven.aux_function import general_aux

# Applies the numerical schemes to the entire domain from t=0 to t=t_end
def entire_domain(out_name, out_dir, bool_plot, x_len, break_pos, g, cells, riemann_solver, riemann_str, tolerance, iterations, t_end, W_l, W_r):
    model = None
    #discritization of the domain
    dx = x_len/cells
    (U,W) = discritization.discretize_initial_values(dx, cells, break_pos, W_l, W_r)
    if riemann_solver == 2:
        model = general_aux.load_model('data_driven/models/riemann_FFNN_shallow.pt', 'cpu', 'ffnn_riemann') # CPU can be changed if one has a Nvidia GPU
    elif riemann_solver == 3:
        model = general_aux.load_model('data_driven/models/godunov_flux_exact_25k.pt', 'cpu', 'godunov_flux') 
    elif riemann_solver == 4:
        model = general_aux.load_model('data_driven/models/godunov_flux_hllc_25k.pt', 'cpu', 'godunov_flux') 
    CFL = 0.9
    t = 0
    end = False
    while t < t_end and not(end):
        #calculate the time step
        if riemann_solver == 2:
            (delta_t, fluxes) = discritization.flux_at_boundaries_data_driven_riemann(W, g, cells, dx, CFL, model)
        if riemann_solver == 3:
            (delta_t, fluxes) = discritization.flux_at_boundaries_data_driven_god(W, g, dx, CFL, model)
        else:
            (delta_t, fluxes) = discritization.flux_at_boundaries(W, g, cells, riemann_solver, dx, tolerance, iterations, CFL)
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
        if riemann_solver == 2:
            plotter.plot(out_name, out_dir, x_len, t_end, cells, (True, False), exact_data, 0, W, riemann_str, True)
        else:
            plotter.plot(out_name, out_dir, x_len, t_end, cells, (True, False), exact_data, 0, W, riemann_str, False)
    return (U, W)

def main(terminal_arguments):
    # reading in terminal arguments
    try:
        read_file = open('input/' + terminal_arguments[1])
    except:
        print('Could not find input file, please remeber to execute the program from the SWE folder and specify the input file as first argument')
        sys.exit(1)
    (x_len, break_pos, g, cells, tolerance, iterations, t_end, h_l, u_l, psi_l, h_r, u_r, psi_r) = file_manipulation.extract(read_file) 

    if not (len(terminal_arguments) == 3):
        print('Please specify the Riemann solver to use as second argument, and provide no more than 2 arguments (input file, and solver)\nSecond argument:\n\'exact\'\n\'hllc\'\n\'data-driven\'')
        sys.exit(1)
    else:
        riemann_str = terminal_arguments[2]
        riemann_solver = 0 # 0 = exact, 1 = HLLC, 2 = data-driven, 3 = flux_from_data_driven
        if (riemann_str == 'exact'):
            pass
        elif (riemann_str == 'hllc'):
            riemann_solver = 1
        elif (riemann_str == 'data-driven'):
            riemann_solver = 2 
        elif (riemann_str == 'data-flux-exact'):
            riemann_solver = 3
        elif (riemann_str == 'data-flux-hllc'):
            riemann_solver = 4
        else:
            print('Please specify exact, hllc, data-driven or data-flux as third argument. To choose the used riemann solver or computing flux entirely by data-driven model')
            sys.exit(1)
    save_dir = "output/numerical/godunov_upwind_results/" + riemann_str
    if riemann_solver == 2:
        save_dir = "output/data_driven/riemann/godunov_upwind_handle_dry/ffnn"
    if riemann_solver == 3 or riemann_solver == 4:
        save_dir = "output/data_driven/data_flux/godunov_upwind/ffnn"
        psi_l, psi_r = 0.0, 0.0
    (_,_) = entire_domain(os.path.splitext(terminal_arguments[1])[0], save_dir, True, x_len, break_pos, g, cells, riemann_solver, riemann_str, tolerance, iterations, t_end, np.array([h_l, u_l, psi_l]), np.array([h_r, u_r, psi_r]))

if __name__ == '__main__':
    main(sys.argv)
    print("Completed successfully Godunov upwind")