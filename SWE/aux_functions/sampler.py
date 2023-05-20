"""The purpose of this class is to be able to sample the solution at a single interface, and the entire domain at a given time t_end. 
We handle both the dry-bed case and the wet-bed case. 
When sampling the entire domain t is a fixed value, and we want to sample the solution over x, which is discretized into a number of points
The number of points is given by the variable cells."""

import math, sys
import numpy as np
from aux_functions import f, exact_riemann_solver, wet_bed

# Sample the at a point s, and return W at that point
def single_sample_wet(g, s, W_l, a_l, h_s, u_s, a_s, W_r, a_r):
    if (s <= u_s): # to the left of the shear wave
        if (h_s > W_l[0]): # the left wave is a shock wave
            q_l = math.sqrt(0.5*((h_s + W_l[0])*h_s)/(W_l[0]**2))
            s_l = W_l[1] - a_l*q_l # the left shock speed 
            if (s <= s_l): # to the left of the left shock
                return W_l
            else: # to the right of the left shock
                return np.array([h_s, u_s, W_l[2].item()])
        else: # the left wave is a rarefaction wave
            s_hl = W_l[1] - a_l # the speed of the head of rarefaction wave
            s_tl = u_s - a_s # the speed of the tail of rarefaction wave
            if(s <= s_hl): # to the left of the rarefaction
                return W_l 
            elif(s <= s_tl): # inside rarefaction wave 
                u_x = (W_l[1]+2*a_l+2*s)/3
                a_x = (W_l[1]+2*a_l-s)/3
                h_x = (a_x**2)/g
                return np.array([h_x, u_x, W_l[2].item()])
            else: # to the right of the rarefaction
                return np.array([h_s, u_s, W_l[2].item()])
    else: # to the right of the shear wave
        if (h_s > W_r[0]): # the right wave is a shock wave
            q_r = math.sqrt(0.5*((h_s + W_r[0])*h_s)/(W_r[0]**2))
            s_r = W_r[1] + a_r*q_r # the right shock speed 
            if (s < s_r): # to the left of the right shock
                return np.array([h_s, u_s, W_r[2].item()])
            else: # to the right of the right shock
                return W_r 
        else: # the right wave is a rarefaction wave
            s_hr = W_r[1] + a_r # the speed of the head of rarefaction wave
            s_tr = u_s + a_s # the speed of the tail of rarefaction wave
            if(s <= s_tr): # to the left of the rarefaction
                return np.array([h_s, u_s, W_r[2].item()])
            elif(s <= s_hr): # inside rarefaction wave 
                u_x = (W_r[1]-2*a_r+2*s)/3
                a_x = (-W_r[1]+2*a_r+s)/3
                h_x = (a_x**2)/g
                return np.array([h_x, u_x, W_r[2].item()])
            else: # to the right of the rarefaction
                return W_r 
    print("Something went completely wrong in the single_sample_wet function")
    sys.exit(1)

# Sample the entire domain in the wet case:
# Retruns a 2D numpy array, first index is cell number 0...cells-1, and second W
def sample_domain_wet(break_pos, x_len, t_end, cells, g, W_l, h_s, u_s, a_s, W_r):
    a_l = math.sqrt(g*W_l[0])
    a_r = math.sqrt(g*W_r[0])
    sol_data = np.empty([cells,3]) 
    for i in range(cells):
        x_i = (i+0.5)*(x_len/cells)-break_pos # moving the break position to x=0
        s = x_i/t_end # the similarity variable
        sol_data[i] = single_sample_wet(g, s, W_l, a_l, h_s, u_s, a_s, W_r, a_r)
    return sol_data

# Sample the at a point s, and return W at that point
def single_sample_dry(g, s, s_sr, s_hr, s_sl, s_hl, W_l, a_l, W_r, a_r):
    if(W_l[0] <= 0): # the left is dry
        if (s <= s_sr): # to the left of the dry/wet front
            return W_l # all these values should be 0
        elif(s <= s_hr): # inside the rarefaction wave
            u_x = (W_r[1]-2*a_r+2*s)/3
            a_x = (-W_r[1]+2*a_r+s)/3
            h_x = (a_x**2)/g
            return np.array([h_x, u_x, W_r[2].item()])
        else: # to the right of the rarefaction
            return W_r 
    elif(W_r[0] <= 0): # the right is dry
        if (s <= s_hl): # to the left of the rarefaction
            return W_l 
        elif(s <= s_sl): # inside the rarefaction wave
            u_x = (W_l[1]+2*a_l+2*s)/3
            a_x = (W_l[1]+2*a_l-s)/3
            h_x = (a_x**2)/g
            return np.array([h_x, u_x, W_l[2].item()])
        else: # to the right of the dry/wet front
            return W_r 
    else: # the dry bed is created in the middel 
        if (s <= s_hl): # to the left of the rarefaction
            return W_l 
        elif (s <= s_sl): # in the left rarefaction
            u_x = (W_l[1]+2*a_l+2*s)/3
            a_x = (W_l[1]+2*a_l-s)/3
            h_x = (a_x**2)/g
            return np.array([h_x, u_x, W_l[2].item()])
        elif (s <= s_sr): # in the dry region
            return np.array([0.0, 0.0, 0.0])
        elif (s <= s_hr): # in the right rarefaction
            u_x = (W_r[1]-2*a_r+2*s)/3
            a_x = (-W_r[1]+2*a_r+s)/3
            h_x = (a_x**2)/g
            return np.array([h_x, u_x, W_r[2].item()])
        else: # to the right of the rarefaction
            return W_r 
    print("something went completely wrong in the single_sample_dry function")

# Sample the entire domain in the dry case:
# Retruns a 2D numpy arary, first index is cell number 0...cells-1, and second is W
def sample_domain_dry(break_pos, x_len, t_end, cells, g, W_l, W_r):
    a_l, a_r = None, None 
    # need this to deal with edge cases for dry bed, where the float value 0 becomes negative, do to inprecision in computers.
    if (W_l[0] <= 0):
        a_l = 0
    else:
        a_l = math.sqrt(g*W_l[0])
    if (W_r[0] <= 0):
        a_r = 0 
    else:
        a_r = math.sqrt(g*W_r[0])
    (s_sr, s_hr, s_sl, s_hl) = f.get_dry_speeds(W_l[1], a_l, W_r[1], a_r)

    sol_data = np.empty([cells, 3]) 
    for i in range(cells):
        x_i = (i+0.5)*(x_len/cells)-break_pos # moving the break position to x=0
        s = x_i/t_end # the similarity variable
        sol_data[i] = single_sample_dry(g, s, s_sr, s_hr, s_sl, s_hl, W_l, a_l, W_r, a_r)
    return sol_data

# sample the entire domain exactly at a given time t_end. 
def sample_exact(break_pos, x_len, t_end, cells, g, W_l, W_r, tolerance, iterations):
    sol_data = None 
    (dry_bool, _, _, _) = exact_riemann_solver.solve(0.0, W_l, W_r, g, tolerance, iterations)
    if dry_bool:
        sol_data = sample_domain_dry(break_pos, x_len, t_end, cells, g, W_l, W_r)
    else:
        (h_s, u_s, a_s) = wet_bed.calculate(g, tolerance, iterations, W_l[0], W_l[1], math.sqrt(g*W_l[0]), W_r[0], W_r[1], math.sqrt(g*W_r[0]))
        sol_data = sample_domain_wet(break_pos, x_len, t_end, cells, g, W_l, h_s, u_s, a_s, W_r)
    return sol_data