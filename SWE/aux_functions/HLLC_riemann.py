# This is the simple HLLC - Riemann solver, which is just Toro (10.21) for the flux of u and h*u, and then extended to psi*u with (10.28).

import sys
import math
from aux_functions import f

# This function gives the flux at boundary between two cells, given the left and right states
def solve(h_l, u_l, psi_l, h_r, u_r, psi_r, g):
    
    a_l, a_r = None, None  
    if (h_l <= 0): # need this to deal with edge cases for dry bed, where the float value 0 becomes negative, do to inprecision in computers.
        a_l = 0
    else:
        a_l = math.sqrt(g*h_l)
    if (h_r <= 0):
        a_r = 0 
    else:
        a_r = math.sqrt(g*h_r)
    
    h_s = 1/2*(h_l+h_r)-1/4*(u_r-u_l)*(h_l+h_r)/(a_l+a_r) # use (10.17) Toro - shock-cap... to approximate h_star 
    u_s = 1/2*(u_l+u_r)-(h_r-h_l)*(a_l+a_r)/(h_l+h_r)
    q_l = f.qk(h_s, h_l)
    q_r = f.qk(h_s, h_r)
    S_l = u_l - a_l*q_l # (10.22) Toro - shock-cap...
    S_r = u_r + a_r*q_r 

    dry_bool = False
    if (h_l <= 0 or h_r <= 0 or h_s <= 0):
        dry_bool = True
    F_l = [h_l*u_l, h_l*u_l*u_l + 1/2*g*h_l*h_l, psi_l*u_l]
    F_r = [h_r*u_r, h_r*u_r*u_r + 1/2*g*h_r*h_r, psi_r*u_r]
    if S_l >= 0:
        return (dry_bool, F_l)
    elif S_l <= 0 and S_r >= 0:
        flux_HLLC_0 = (S_r*F_l[0] - S_l*F_r[0] + S_l*S_r*(h_r-h_l))/(S_r-S_l) 
        flux_HLLC_1 = (S_r*F_l[1] - S_l*F_r[1] + S_l*S_r*(h_r*u_r-h_l*u_l))/(S_r-S_l)
        if u_s >= 0: # 10.28 
            flux_HLLC_2 = flux_HLLC_0 * psi_l
            return (dry_bool, [flux_HLLC_0, flux_HLLC_1, flux_HLLC_2])
        else:
            flux_HLLC_2 = flux_HLLC_0 * psi_r
            return (dry_bool, [flux_HLLC_0, flux_HLLC_1, flux_HLLC_2])
    elif S_r <= 0:
        return (dry_bool, F_r)
    print("This should never happen, something went wrong in HLLC_riemann.py") 
    sys.exit(1)

    
    