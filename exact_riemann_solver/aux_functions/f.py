"""
This implements the functions f, f_l and f_r given in Toro - page 96-97
And the differential of f, f_l and f_r given in Toro - page 98
"""

import math

# calculate the function f_k, where k is the left or right side of the interface
def f_k(g, h, h_k):
    #rarefaction left
    if h <= h_k:
        return 2*(math.sqrt(h*g)-math.sqrt(h_k*g))
    #shock left
    else:
        return (h-h_k) * math.sqrt(1/2*g*((h+h_k)/(h*h_k)))

# calculate the function f
def f(g, h, h_l, h_r, u_l, u_r):
    return f_k(g, h, h_l) + f_k(g, h,h_r)+u_r-u_l

# calculate the derivative of f_k
def fkd(g, h_s, h_k, a_k):
    
    f_k, f_kd = 0, 0

    # two cases: 1. rarefaction wave
    if (h_s <= h_k):
        f_kd = g/a_k 
    else:   # 2. shock wave
        g_k = math.sqrt(1/2*g*((h_s+h_k)/(h_s*h_k)))
        f_kd = g_k-(g*(h_s-h_k)/(4*(h_s**2)*g_k)) # also (5.13) second part
    return f_kd
        
 