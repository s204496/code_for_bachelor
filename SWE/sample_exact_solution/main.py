""" 
This algorithm solves the exact Riemann problem for the dam-break problem with both wet and dry bed. 
And is heavily inspired by the algorithm given in Toro - Shock-capturing methods for free-surface flows - page 128-140.

This is the main program that calls the other functions, it reads in the input from a file in ./input.
Further it writes the output to a file in ./data.
As terminal arguments you should specify input file  and output file. Like ```python exact_Riemann_solver/main.py input.txt output.txt```

The input should state the following hyperparameters:

1. x-length: the length of the domain in the x-direction, the domain is always [0,x-length] 
2. break-position: the position along the x-axis where we have the discountinues change in h and u at t=0
3. gravity: the gravity constant, default is 9.8
4. cells: number of points that we want the solution evaluated at.
5. tolerance: to what precision do we want the exact solution to the dam-break problem
6. iterations: maximum number of iterations for the solver, when using Newton-Raphson
7. t_end: the t value, that we want a solution for, then at this time t, we can sample x.  
8. h_l: height of on the left of the break-position at t=0.
9. u_l: speed of on the left of the break-position at t=0, positive u is to the right and negative to the left.
10. psi_l: the value of the polutent on the left of the break-position at t=0.
11. h_r: height of on the right of the break-position at t=0.
12. u_r: speed of on the right of the break-position at t=0, positive u is to the right and negative to the right.
13. psi_r: the value of the polutent on the right of the break-position at t=0.
"""

# Imports
import sys
import os
sys.path.append('../SWE')
import math
from aux_functions import sampler, plotter, file_manipulation

def main(terminal_arguments):
    #reading in values
    try:
        read_file = open('input/' + terminal_arguments[1])
    except:
        print('Could not find input file, please remeber to execute the program from the root folder and specify the input file as first argument')
        sys.exit(1)
    (x_len, break_pos, g, cells, tolerance, iterations, t_end, h_l, u_l, psi_l, h_r, u_r, psi_r) = file_manipulation.extract(read_file) 

    #open the file writen to
    try:
        #this creates the file if it does not exist, and overwrites it if it does
        out_file = open('output/exact_solutions/' + terminal_arguments[2], 'w')
    except:
        print('Could not find output file, please specify the output file as second argument')
        sys.exit(1)

    sol_data = sampler.sample_exact(True, out_file, break_pos, x_len, t_end, cells, g, h_l, u_l, psi_l, h_r, u_r, psi_r, tolerance, iterations)
    plotter.plot(os.path.splitext(terminal_arguments[2])[0], 'output/exact_solutions', False, True, x_len, t_end, cells, (True, True), sol_data, 0, "not used", "not used")

if __name__ == '__main__':
    main(sys.argv)
    print("Completed successfully")