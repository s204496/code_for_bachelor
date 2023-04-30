""" This function extracts the content of a testcase file.
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

def extract(input_file):
    x_len = float(input_file.readline().split()[1])
    break_pos = float(input_file.readline().split()[1])
    g = float(input_file.readline().split()[1])
    cells = int(input_file.readline().split()[1])
    tolerance = float(input_file.readline().split()[1])
    iterations = int(input_file.readline().split()[1])
    t_end = float(input_file.readline().split()[1])
    h_l = float(input_file.readline().split()[1])
    u_l = float(input_file.readline().split()[1])
    psi_l = float(input_file.readline().split()[1])
    h_r = float(input_file.readline().split()[1])
    u_r = float(input_file.readline().split()[1])
    psi_r = float(input_file.readline().split()[1])
    return (x_len, break_pos, g, cells, tolerance, iterations, t_end, h_l, u_l, psi_l, h_r, u_r, psi_r)