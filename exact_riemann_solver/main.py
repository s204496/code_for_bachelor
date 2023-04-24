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
6. iteration: maximum number of iterations for the solver, when using Newton-Raphson
7. t_end: the t value, that we want a solution for, then at this time t, we can sample x.  
8. h_l: height of on the left of the break-position at t=0.
9. u_l: speed of on the left of the break-position at t=0, positive u is to the right and negative to the left.
10. h_r: height of on the right of the break-position at t=0.
11. u_r: speed of on the right of the break-position at t=0, positive u is to the right and negative to the right.
"""

# Imports
import sys
import os
sys.path.append('exact_riemann_solver')
import math
from aux_functions import wet_bed, sampler, plotter

def main(terminal_arguments):
    #reading in values
    try:
        read_file = open('exact_riemann_solver/input/' + terminal_arguments[1])
    except:
        print('Could not find input file, please remeber to execute the program from the root folder and specify the input file as first argument')
        sys.exit(1)
    x_len = float(read_file.readline().split()[1])
    break_pos = float(read_file.readline().split()[1])
    g = float(read_file.readline().split()[1])
    cells = int(read_file.readline().split()[1])
    tolerance = float(read_file.readline().split()[1])
    iteration = int(read_file.readline().split()[1])
    t_end = float(read_file.readline().split()[1])
    h_l = float(read_file.readline().split()[1])
    u_l = float(read_file.readline().split()[1])
    h_r = float(read_file.readline().split()[1])
    u_r = float(read_file.readline().split()[1])

    #open the file writen to
    try:
        #this creates the file if it does not exist, and overwrites it if it does
        out_file = open('exact_riemann_solver/output/' + terminal_arguments[2], 'w')
    except:
        print('Could not find output file, please specify the output file as second argument')
        sys.exit(1)

    #computing celerity on the left and right side
    a_l = math.sqrt(g*h_l)
    a_r = math.sqrt(g*h_r)

    # we check whether the depth posittivity condition is satisfied, you can see this condition in Toro - Shock-cap... - page 100
    dpc = 2*(a_l + a_r) >= (u_r - u_l)

    # Dry bed case
    if (not(dpc) or h_l <= 0 or h_r <= 0):
        out_file.write('Case: Dry bed\n')
        sol_data = sampler.sample_dry(out_file, x_len, break_pos, t_end, cells, g, h_l, h_r, u_l, u_r, a_l, a_r)
        plotter.plot(os.path.splitext(terminal_arguments[2])[0], sol_data, x_len, break_pos, t_end, cells)
    # Wet bed 
    else:
        out_file.write('Case: Wet bed\n')
        (h_s, u_s, a_s) = wet_bed.calculate(out_file, g, tolerance, iteration, h_l, h_r, u_l, u_r, a_l, a_r)
        sol_data = sampler.sample_wet(out_file, x_len, break_pos, t_end, cells, g, h_l, h_s, h_r, u_l, u_s, u_r, a_l, a_s, a_r)
        plotter.plot(os.path.splitext(terminal_arguments[2])[0], sol_data, x_len, break_pos, t_end, cells)

if __name__ == '__main__':
    main(sys.argv)