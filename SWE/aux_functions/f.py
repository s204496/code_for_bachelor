"""
This implements the functions f, f_l and f_r given in Toro - page 96-97
And the differential of f, f_l and f_r given in Toro - page 98
"""

import math
import numpy as np

# calculate the function f_k, where k is the left or right side of the interface
def f_k(g, h, h_k):
    #rarefaction left
    if h <= h_k:
        return 2*(math.sqrt(h*g)-math.sqrt(h_k*g))
    #shock left
    else:
        if (h < 10e-8 or h_k < 10e-8): # need this to take care of edge case where h or h_k is very small
            return 0.0
        else:
            return (h-h_k) * math.sqrt(1/2*g*((h+h_k)/(h*h_k)))

# calculate the function f
def f(g, h, h_l, u_l, h_r, u_r):
    return f_k(g, h, h_l) + f_k(g, h,h_r)+u_r-u_l

# calculate the derivative of f_k
def fkd(g, h_s, h_k, a_k):
    
    f_k, f_kd = 0, 0

    # two cases: 1. rarefaction wave
    if (h_s <= h_k):
        f_kd = g/a_k 
    else:   # 2. shock wave
        if (h_s < 10e-8 or h_k < 10e-8): # need this to take care of edge case where h or h_k is very small
            return 0.0
        else:
            g_k = math.sqrt(1/2*g*((h_s+h_k)/(h_s*h_k)))
            f_kd = g_k-(g*(h_s-h_k)/(4*(h_s**2)*g_k)) # also (5.13) second part
    return f_kd
        
# get the speed of the dry/wet waves all four cases
def get_dry_speeds(u_l, a_l, u_r, a_r):
    return (u_r - 2*a_r, u_r + a_r, u_l + 2*a_l, u_l - a_l) # dry/wet front right, head right rarefaction, dry/wet front left, head left rarefaction
 
def qk(h_s, h_k): # this function is Toro (10.23)
    if (h_s > h_k):
        return math.sqrt(1/2*(h_s+h_k)*h_s/(h_k**2))
    else:
        return 1.0

def flux_from_w(W, g):
    return np.array([W[0]*W[1], W[0]*(W[1]**2) + 0.5*g*(W[0]**2), W[0]*W[1]*W[2]])