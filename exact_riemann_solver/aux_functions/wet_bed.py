"""purpose of this function is to solve the Riemann problem, to the given precision for the wet bed case"""

import math
from aux_functions import newton_raphson, f

def calculate(out_file, x_len, break_pos, g, cells, tolerance, iterations, t_end, h_l, u_l, h_r, u_r, a_l, a_r):
    ### get an initial guess
    h_0 = newton_raphson.initial_guess(g, h_l, h_r, u_l, u_r, a_l, a_r)
    out_file.write("Initial guess h_0 in the mid region: " + str(h_0) + "\n")
    h_s = newton_raphson.newton_rapson_iter(out_file, g, h_0, h_l, h_r, a_l, a_r, u_l, u_r, tolerance, iterations)
    a_s = math.sqrt(g*h_s)
    f_l = f.f_k(g, h_s, h_l)
    f_r = f.f_k(g, h_s, h_r)
    u_s = 0.5*(u_l+u_r)+0.5*(f_r-f_l) # using the closed form given in (5.8) - Toro - Shock-cap... p.97
    out_file.write("Depth of star region: " + str(h_s) + "\n")
    out_file.write("Velocity of star region: " + str(u_s) + "\n")

    ### sample the solution in at time t_end 
