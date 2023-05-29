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

# Get a single sample returning the time step and result of a single time-step
def single_sample(solver, tolerance, iterations, dx, cfl, g, W, U):
    (dt, fluxes) = discritization.flux_at_boundaries(W, g, 3, solver, dx, tolerance, iterations, cfl)
    discritization.evolve(U, fluxes, dx, dt, 3)
    return (dt, U[2])

# Applies the numerical schemes to the entire domain from t=0 to t=t_end
def entire_domain(out_name, out_dir, bool_plot, x_len, break_pos, g, cells, riemann_int, riemann_str, tolerance, iterations, t_end, W_l, W_r, data_driven):
    model = None
    if data_driven == 1:
        if riemann_int == 0:
            model = general_aux.load_model('data_driven/models/riemann_exact_CNN.pt', 'cpu', 'cnn_riemann') # CPU can be changed if one has a Nvidia GPU
        else:
            model = general_aux.load_model('data_driven/models/riemann_hllc_CNN.pt', 'cpu', 'cnn_riemann')
    elif data_driven == 2:
        if riemann_int == 0:
            model = general_aux.load_model('data_driven/models/riemann_exact_FFNN.pt', 'cpu', 'ffnn_riemann') # CPU can be changed if one has a Nvidia GPU
        else:
            model = general_aux.load_model('data_driven/models/riemann_hllc_FFNN.pt', 'cpu', 'ffnn_riemann')
    #discritization of the domain
    dx = x_len/cells
    (U,W) = discritization.discretize_initial_values(dx, cells, break_pos, W_l, W_r)
    CFL = 0.9
    t = 0
    end = False
    while t < t_end and not(end):
        #calculate the time step
        if (data_driven):
            (delta_t, fluxes) = discritization.flux_at_boundaries_data_driven(W, g, cells, dx, CFL, model, data_driven)
        else:
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
        plotter.plot(out_name, out_dir, x_len, t_end, cells, (True, False), exact_data, 0, W, riemann_str, data_driven)
    return (U, W)

def main(terminal_arguments):
    # reading in terminal arguments
    try:
        read_file = open('input/' + terminal_arguments[1])
    except:
        print('Could not find input file, please remeber to execute the program from the root folder and specify the input file as first argument')
        sys.exit(1)
    (x_len, break_pos, g, cells, tolerance, iterations, t_end, h_l, u_l, psi_l, h_r, u_r, psi_r) = file_manipulation.extract(read_file) 

    if not (len(terminal_arguments) == 4):
        print('Please specify the Riemann solver to use as second argument and whether or not data driven model should be used. \n Write \'exact\' or \'hllc\' For the solver \n for data driven model write \'0\' for schemes, \'1\' for datadriven CNN and \'2\' for FFNN, and provide no more than 3 arguments')
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
    data_driven = 0 
    save_dir = "output/numerical/godunov_upwind_results/" + riemann_str
    if (terminal_arguments[3] == '2'):
        data_driven = 2 
        save_dir = "output/data_driven_riemann/godunov_upwind/ffnn/"
    elif (terminal_arguments[3] == '1'):
        data_driven = 1 
        save_dir = "output/data_driven_riemann/godunov_upwind/cnn/"
    elif not(terminal_arguments[3] == '0'):
        print('Please specify \'0\', \'1\' or \'2\' as third argument. To choose whether or not to use the data driven model\n0: no data driven model\n1: CNN\n2: FFNN')
        sys.exit(1)
    (_,_) = entire_domain(os.path.splitext(terminal_arguments[1])[0], save_dir, True, x_len, break_pos, g, cells, riemann_int, riemann_str, tolerance, iterations, t_end, np.array([h_l, u_l, psi_l]), np.array([h_r, u_r, psi_r]), data_driven)

if __name__ == '__main__':
    main(sys.argv)
    print("Completed successfully Godunov upwind")