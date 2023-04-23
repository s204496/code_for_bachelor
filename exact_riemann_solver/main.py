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
sys.path.append('exact_riemann_solver')
import math
from aux_functions import dry_bed, wet_bed, sampler

def print_variables(x_len, break_pos, g, cells, tolerance, iteration, t_end, h_l, u_l, h_r, u_r):
    print('x-length: ', str(x_len))
    print('break-position: ', str(break_pos))
    print('gravity: ', str(g))
    print('cells: ', str(cells))
    print('tolerance: ', str(tolerance))
    print('iteration: ', str(iteration))
    print('t_end: ', str(t_end))
    print('h_l: ', str(h_l))
    print('u_l: ', str(u_l))
    print('h_r: ', str(h_r))
    print('u_r: ', str(u_r))


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
        out_file = open('exact_riemann_solver/data/' + terminal_arguments[2], 'w')
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
        value = dry_bed.calculate(out_file, x_len, break_pos, g, cells, tolerance, iteration, t_end, h_l, u_l, h_r, u_r, a_l, a_r)
        print(value)
    # Wet bed 
    else:
        ####!!!!
        out_file.write('Case: Wet bed\n')
        value = wet_bed.calculate(out_file, x_len, break_pos, g, cells, tolerance, iteration, t_end, h_l, u_l, h_r, u_r, a_l, a_r)
        print(value)
        print("wet bed")
    
    sampler.sample(out_file, value)
    print_variables(x_len, break_pos, g, cells, tolerance, iteration, t_end, h_l, u_l, h_r, u_r)
    

if __name__ == '__main__':
    main(sys.argv)