# This class runs multiple instances of the Godunov scheme, and plots the results of accuracy and speed

import sys
import os
sys.path.append('../SWE')
import time
import godunov_upwind
import numpy as np
from aux_functions import sampler, error_calculation, file_manipulation, plotter

def main(terminal_arguments):
    try:
        read_file = open('input/' + terminal_arguments[1])
    except:
        print('Could not find input file, please remeber to execute the program from the root folder and specify the input file as first argument')
        sys.exit(1)
    (x_len, break_pos, g, _, tolerance, iterations, t_end, h_l, u_l, psi_l, h_r, u_r, psi_r) = file_manipulation.extract(read_file) 

    if not (len(terminal_arguments) == 3):
        print('Please specify the Riemann solver to use as second argument, and provide no more than 2 arguments (input file, and solver)')
        sys.exit(1)
    else:
        riemann_str = terminal_arguments[2]
        riemann_int = 0 # 0 = exact, 1 = HLLC
        if (riemann_str == 'exact'):
            pass
        elif (riemann_str == 'hllc'):
            riemann_int = 1
        else:
            print('Please specify exact or hllc as third argument. To choose the used riemann solver')
            sys.exit(1)
    
    cells_list = [100*2**i for i in range(7)]
    delta_x_list = [x_len/cells for cells in cells_list] 
    W_l, W_r = np.array([h_l, u_l, psi_l]), np.array([h_r, u_r, psi_r])
    error_list = [[],[],[]]
    speed_list = []

    for cells in cells_list:
        start_time = time.time()
        (U,W) = godunov_upwind.entire_domain("", "", False, x_len, break_pos, g, cells, riemann_int, riemann_str, tolerance, iterations, t_end, W_l, W_r)
        end_time = time.time()
        elapsed_time = end_time - start_time
        speed_list.append(elapsed_time)
        exact = sampler.sample_exact(break_pos, x_len, t_end, cells, g, W_l, W_r, tolerance, iterations)
        for i in range(3):
            error =  error_calculation.norm_2_fvm(exact[:, i], W[1:-1, i], cells, x_len)
            if (error < 0.0000001):
                error_list[i].append(0.00001)
            else:
                error_list[i].append(error)
        print('completed cells: ' + str(cells))
    
    plotter.plot_error_and_speed(speed_list, error_list, delta_x_list, cells_list, os.path.splitext(terminal_arguments[1])[0], "output/speed_and_accuracy/godunov_upwind", "godunov_upwind", riemann_str)

if __name__ == '__main__':
    main(sys.argv)
    print("Completed successfully")