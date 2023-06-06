
import sys
import os
import numpy as np
sys.path.append('../SWE')
import math
from aux_functions import sampler, plotter, file_manipulation

""" 
This is an exact solution to the dam-break problems given in test 1-5. These are taken directly from Toro's book.
"""
def main(terminal_arguments):
    #reading in values
    try:
        read_file = open('input/' + terminal_arguments[1])
    except:
        print('Could not find input file, please remeber to execute the program from the root folder and specify the input file as first argument')
        sys.exit(1)
    (x_len, break_pos, g, cells, tolerance, iterations, t_end, h_l, u_l, psi_l, h_r, u_r, psi_r) = file_manipulation.extract(read_file) 

    sol_data = sampler.sample_exact(break_pos, x_len, t_end, cells, g, np.array([h_l, u_l, psi_l]), np.array([h_r, u_r, psi_r]), tolerance, iterations)
    plotter.plot(os.path.splitext(terminal_arguments[1])[0], 'output/exact_solutions', x_len, t_end, cells, (True, True), sol_data, -1, np.array([]), "", 0)

if __name__ == '__main__':
    main(sys.argv)
    print("Completed successfully")