# This is the simple HLLC - Riemann solver, which is just (Toro (10.21) - page 180) for the flux of u and h*u, and then extended to psi*u with (10.28).

import sys
import math
import numpy as np
from aux_functions import f

# just a aux function for the function below
def compute_w_rarefaction(left_rarefaction, g, W_k):
    h, u = None, None
    if (left_rarefaction):
        u = 1/3*(W_k[1]+2*math.sqrt(g*W_k[0]))
        h = (1/3*(W_k[1]+2*math.sqrt(g*W_k[0])))**2/g
    else:
        u = 1/3*(W_k[1]-2*math.sqrt(g*W_k[0]))
        h = (1/3*(-W_k[1]+2*math.sqrt(g*W_k[0])))**2/g
    return np.array([h, u, W_k[2].item()])

""" 
This Riemann solver and gives return a quadruple of 
1: A bool describing, whether or not one of the sides or the star region is dry
2: The h, u, psi at the given x/t=0. This is a numpy array of floats
3: The Flux at the given x/t=0. This is calculated from W, and is a numpy array of 3 floats
4: A tuple of floats, for h and u in the star region, these are set to 0.0, whenever there is a dry/wet front. 
"""
# This function gives the flux at boundary between two cells, given the left and right states
def solve(W_l, W_r, g):

    if (W_l[0] <= 0): # left dry
        if (W_r[0] <= 0):
            return (True, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), (0.0, 0.0))
        W = compute_w_rarefaction(False, g, W_r)
        return (True, W, f.flux_from_w(W, g), (0.0, 0.0))
    elif (W_r[0] <= 0): # right dry
        W = compute_w_rarefaction(True, g, W_l)
        return (True, W, f.flux_from_w(W, g), (0.0, 0.0))

    a_l = math.sqrt(g*W_l[0])
    a_r = math.sqrt(g*W_r[0])
    
    #h_s = 1/2*(W_l[0]+W_r[0])-1/4*(W_r[1]-W_l[1])*(W_l[0]+W_r[0])/(a_l+a_r) # use (10.17) Toro - shock-cap... to approximate h_star 
    h_s = 1/g*((1/2*(a_l+a_r)+1/4*(W_l[1]-W_r[1]))**2) # use (10.18) Toro - shock-cap... to approximate h_star 
    if(h_s <= 0): # created dry bed
        return (True, np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), (0.0, 0.0))
    #u_s = 1/2*(W_l[1]+W_r[1])-(W_r[0]-W_l[0])*(a_l+a_r)/(W_l[0]+W_r[0]) # (10.17)
    u_s = 1/2*(W_l[1]+W_r[1])+a_l-a_r # (10.18)
    if (u_s > 0 and h_s <= W_l[0] and not(np.sign(W_l[1] - a_l) == np.sign(u_s-math.sqrt(h_s*g)))): 
    #This is the case, where we are inside a left rarefaction wave, the HLLC riemann cannot be used as described by Toro, we take care of this edge case.
        W = compute_w_rarefaction(True, g, W_l)
        return (False, W, f.flux_from_w(W, g), (h_s, u_s)) 
    elif (u_s < 0 and h_s <= W_r[0] and not(np.sign(W_r[1] + a_r) == np.sign(u_s+math.sqrt(h_s*g)))):
        W = compute_w_rarefaction(False, g, W_r)
        return (False, W, f.flux_from_w(W, g), (h_s, u_s)) 
    q_l = f.qk(h_s, W_l[0])
    q_r = f.qk(h_s, W_r[0])
    S_l = W_l[1] - a_l*q_l # (Toro (10.22) Toro - page 180)
    S_r = W_r[1] + a_r*q_r 

    F_l = f.flux_from_w(W_l, g)
    F_r = f.flux_from_w(W_r, g) 
    if S_l >= 0:
        return (False, W_l, F_l, (h_s, u_s))
    elif S_r <= 0:
        return (False, W_r, F_r, (h_s, u_s))
    elif S_l <= 0 and S_r >= 0:
        flux_HLLC_0 = (S_r*F_l[0] - S_l*F_r[0] + S_l*S_r*(W_r[0]-W_l[0]))/(S_r-S_l) 
        flux_HLLC_1 = (S_r*F_l[1] - S_l*F_r[1] + S_l*S_r*(W_r[0]*W_r[1]-W_l[0]*W_l[1]))/(S_r-S_l)
        if u_s >= 0: # 10.28 
            flux_HLLC_2 = flux_HLLC_0 * W_l[2]
            return (False, np.array([h_s, u_s, W_l[2].item()]), np.array([flux_HLLC_0, flux_HLLC_1, flux_HLLC_2]), (h_s, u_s))
        else:
            flux_HLLC_2 = flux_HLLC_0 * W_r[2]
            return (False, np.array([h_s, u_s, W_r[2].item()]), np.array([flux_HLLC_0, flux_HLLC_1, flux_HLLC_2]), (h_s, u_s))
    print("This should never happen, something went wrong in HLLC_riemann.py") 
    sys.exit(1)

    
    