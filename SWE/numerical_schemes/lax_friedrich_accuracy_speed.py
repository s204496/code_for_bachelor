# This class runs multiple instances of the lax_friedrich scheme, and plots the results of accuracy and speed

import sys
import os
sys.path.append('../SWE')
import time
import lax_friedrich 
import numpy as np
from aux_functions import sampler, error_calculation, file_manipulation, plotter

def main(terminal_arguments):
    try:
        read_file = open('input/' + terminal_arguments[1])
    except:
        print('Could not find input file, please remeber to execute the program from the root folder and specify the input file as first argument')
        sys.exit(1)
    (x_len, break_pos, g, _, tolerance, iterations, t_end, h_l, u_l, psi_l, h_r, u_r, psi_r) = file_manipulation.extract(read_file) 

    cells_list = [100*2**i for i in range(7)]
    delta_x_list = [x_len/cells for cells in cells_list] 
    W_l, W_r = np.array([h_l, u_l, psi_l]), np.array([h_r, u_r, psi_r])
    error_list = [[],[],[]]
    speed_list = []

    for cells in cells_list:
        start_time = time.time()
        (U,W) = lax_friedrich.entire_domain("", "", False, x_len, break_pos, g, cells, tolerance, iterations, t_end, W_l, W_r)
        end_time = time.time()
        elapsed_time = end_time - start_time
        speed_list.append(elapsed_time)
        exact = sampler.sample_exact(break_pos, x_len, t_end, cells, g, W_l, W_r, tolerance, iterations)
        for i in range(3):
            error =  error_calculation.norm_2_fvm(exact[:, i], W[1:-1, i], cells, x_len)
            error_list[i].append(error)
        print('completed cells: ' + str(cells))
    
    plotter.plot_error_and_speed(speed_list, error_list, delta_x_list, cells_list, os.path.splitext(terminal_arguments[1])[0], "output/speed_and_accuracy/lax_friedrich", "lax_friedrich", "")

if __name__ == '__main__':
    main(sys.argv)
    print("Completed successfully")