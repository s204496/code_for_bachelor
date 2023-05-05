# This is the simple HLLC - Riemann solver, which is just Toro (10.21) for the flux of u and h*u, and then extended to psi*u with (10.28).

import sys
import math
import numpy as np
from aux_functions import f

def compute_flux_rarefaction_interface(left_rarefaction, g, h_k, u_k, psi_k):
    h, u = None, None
    if (left_rarefaction):
        u = 1/3*(u_k+2*math.sqrt(g*h_k))
        h = (1/3*(u_k+2*math.sqrt(g*h_k)))**2/g
    else:
        u = 1/3*(u_k-2*math.sqrt(g*h_k))
        h = (1/3*(-u_k+2*math.sqrt(g*h_k)))**2/g
    flux_HLLC_0 = h*u
    flux_HLLC_1 = h*(u**2)+1/2*g*h**2
    flux_HHLC_2 = h*u*psi_k
    return [flux_HLLC_0, flux_HLLC_1, flux_HHLC_2]

# This function gives the flux at boundary between two cells, given the left and right states
def solve(h_l, u_l, psi_l, h_r, u_r, psi_r, g):

    if (h_l <= 0): # left dry
        if (h_r <= 0):
            return [True, [0.0, 0.0, 0.0]]
        return [True, compute_flux_rarefaction_interface(False, g, h_r, u_r, psi_r), [0.0, 0.0]]
    elif (h_r <= 0): # right dry
        if (h_l <= 0):
            return (True, [0.0, 0.0, 0.0])
        return [True, compute_flux_rarefaction_interface(True, g, h_l, u_l, psi_l), [0.0, 0.0]]

    a_l = math.sqrt(g*h_l)
    a_r = math.sqrt(g*h_r)
    
    #h_s = 1/2*(h_l+h_r)-1/4*(u_r-u_l)*(h_l+h_r)/(a_l+a_r) # use (10.17) Toro - shock-cap... to approximate h_star 
    h_s = 1/g*((1/2*(a_l+a_r)+1/4*(u_l-u_r))**2) # use (10.18) Toro - shock-cap... to approximate h_star 
    if(h_s <= 0): # created dry bed
        return [True, [0.0, 0.0, 0.0], [0.0, 0.0]]
    #u_s = 1/2*(u_l+u_r)-(h_r-h_l)*(a_l+a_r)/(h_l+h_r) # (10.17)
    u_s = 1/2*(u_l+u_r)+a_l-a_r # (10.18)
    if (u_s > 0 and h_s <= h_l and not(np.sign(u_l - a_l) == np.sign(u_s-math.sqrt(h_s*g)))): 
    #This is the case, where we are inside a left rarefaction wave, the HLLC riemann cannot be used as described by Toro, we take care of this edge case.
        return [False, compute_flux_rarefaction_interface(True, g, h_l, u_l, psi_l), [h_s, u_s]] 
    elif (u_s < 0 and h_s <= h_r and not(np.sign(u_r + a_r) == np.sign(u_s+math.sqrt(h_s*g)))):
        return [False, compute_flux_rarefaction_interface(False, g, h_r, u_r, psi_r), [h_s, u_s]] 
    q_l = f.qk(h_s, h_l)
    q_r = f.qk(h_s, h_r)
    S_l = u_l - a_l*q_l # (10.22) Toro - shock-cap...
    S_r = u_r + a_r*q_r 

    F_l = [h_l*u_l, h_l*u_l*u_l + 1/2*g*h_l*h_l, h_l*psi_l*u_l]
    F_r = [h_r*u_r, h_r*u_r*u_r + 1/2*g*h_r*h_r, h_r*psi_r*u_r]
    if S_l >= 0:
        return [False, F_l, [h_s, u_s]]
    elif S_r <= 0:
        return [False, F_r, [h_s, u_s]]
    elif S_l <= 0 and S_r >= 0:
        flux_HLLC_0 = (S_r*F_l[0] - S_l*F_r[0] + S_l*S_r*(h_r-h_l))/(S_r-S_l) 
        flux_HLLC_1 = (S_r*F_l[1] - S_l*F_r[1] + S_l*S_r*(h_r*u_r-h_l*u_l))/(S_r-S_l)
        if u_s >= 0: # 10.28 
            flux_HLLC_2 = flux_HLLC_0 * psi_l
            return [False, [flux_HLLC_0, flux_HLLC_1, flux_HLLC_2], [h_s, u_s]]
        else:
            flux_HLLC_2 = flux_HLLC_0 * psi_r
            return [False, [flux_HLLC_0, flux_HLLC_1, flux_HLLC_2], [h_s, u_s]]
    print("This should never happen, something went wrong in HLLC_riemann.py") 
    sys.exit(1)

    
    