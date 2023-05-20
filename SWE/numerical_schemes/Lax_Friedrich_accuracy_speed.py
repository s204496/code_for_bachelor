# This class runs multiple instances of the lax_friedrich scheme, and plots the results of accuracy and speed

import sys
import os
sys.path.append('../SWE')
import time
import Lax_Friedrich 
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
    h_list = [1/cells for cells in cells_list] 
    W_l, W_r = np.array([h_l, u_l, psi_l]), np.array([h_r, u_r, psi_r])
    error_list = []
    speed_list = []

    for cells in cells_list:
        start_time = time.time()
        (U,W) = Lax_Friedrich.lax_friedrich(False, "", "", "", False, x_len, break_pos, g, cells, tolerance, iterations, t_end, W_l, W_r)
        end_time = time.time()
        elapsed_time = end_time - start_time
        speed_list.append(elapsed_time)
        exact_np = sampler.sample_exact(break_pos, x_len, t_end, cells, g, W_l, W_r, tolerance, iterations)
        W_np = np.array(W) #num_r = numerical result 
        error =  error_calculation.norm_2_FVM(exact_np[:, :], W_np[1:-1, :], cells)
        error_list.append(error)
    
    plotter.plot_error_and_speed(speed_list, error_list, h_list, cells_list, os.path.splitext(terminal_arguments[1])[0], "output/speed_and_accuracy/Lax Friedrich", "Lax Friedrich", "no riemann solver")

if __name__ == '__main__':
    main(sys.argv)
    print("Completed successfully")