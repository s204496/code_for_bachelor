import sys
import math
from aux_functions import f, wet_bed, sampler

""" 
This Riemann solver takes in a single point s_t_ratio, usually 0, and gives return a quadruple of 
1: A bool describing, whether or not one of the sides or the star region is dry
2: The h, u, psi at the given s_t_ratio, remeber similarty solution along x/t. This is a numpy array of floats
3: The Flux at the given s_t_ratio, this is calculated from the h, u and psi variables. This is a numpy array of 3 floats
4: A tuple of floats, for h and u in the star region, these are set to 0.0, whenever there is a dry/wet front. 
"""
 
def solve(s_t_ratio, W_l, W_r, g, tolerance, iteration):
    
    #computing celerity on the left and right side
    a_l, a_r = None, None 
    if (W_l[0] <= 0): # need this to deal with edge cases for dry bed, where the float value 0 becomes negative, do to inprecision in computers.
        a_l = 0
    else:
        a_l = math.sqrt(g*W_l[0])
    if (W_r[0] <= 0):
        a_r = 0 
    else:
        a_r = math.sqrt(g*W_r[0])
    
    # we check whether the depth posittivity condition is satisfied, you can see this condition in Toro - Shock-cap... - page 100
    dpc = 2*(a_l + a_r) > (W_r[1] - W_l[1])

    # Dry bed case
    if (not(dpc) or W_l[0] <= 0 or W_r[0] <= 0):
        (s_sr, s_hr, s_sl, s_hl) = f.get_dry_speeds(W_l[1], a_l, W_r[1], a_r)
        W_x = sampler.single_sample_dry(g, s_t_ratio, s_sr, s_hr, s_sl, s_hl, W_l, a_l, W_r, a_r)
        return (True, W_x, f.flux_from_w(W_x, g), (0.0 ,0.0)) # no star region means h_s = 0.0, u_s = 0.0
    else: # Wet bed case   
        (h_s, u_s, a_s) = wet_bed.calculate(g, tolerance, iteration, W_l[0], W_l[1], a_l, W_r[0], W_r[1], a_r)
        W_x = sampler.single_sample_wet(g, s_t_ratio, W_l, a_l, h_s, u_s, a_s, W_r, a_r)
        return (False, W_x, f.flux_from_w(W_x, g), (h_s, u_s))
    