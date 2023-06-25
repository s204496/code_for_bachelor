import sys
import os
import numpy as np
sys.path.append('../SWE')
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
    (x_len, break_pos, _, cells, _, _, t_end, h_l, u_l, psi_l, h_r, u_r, psi_r) = file_manipulation.extract(read_file) 

    plotter.plot_initial(os.path.splitext(terminal_arguments[1])[0], 'output/exact_solutions/initial', x_len, break_pos, t_end, cells, h_l, h_r, u_l, u_r, psi_l, psi_r)

if __name__ == '__main__':
    main(sys.argv)
    print("Completed successfully")