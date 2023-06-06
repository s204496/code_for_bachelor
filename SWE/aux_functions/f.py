"""
This implements the functions f, f_l and f_r given in Toro - page 96-97
And the differential of f, f_l and f_r given in Toro - page 98
Further there are auxiliary functions for finding:
1. q_k (10.23).
2. All possible wave speeds, when the is a dry/wet front.
3. Calculating the flux from W
"""
import torch
import torch.nn as nn
import math
import numpy as np

# Calculate the function f
def f(g, h, h_l, u_l, h_r, u_r):
    return f_k(g, h, h_l) + f_k(g, h,h_r)+u_r-u_l

# Calculate the function f_k, where k is the left or right side of the interface
def f_k(g, h, h_k):
    #rarefaction
    if h <= h_k:
        return 2*(math.sqrt(h*g)-math.sqrt(h_k*g))
    #shock
    else:
        if (h < 10e-8 or h_k < 10e-8): # need this to take care of edge case where h or h_k is very small
            return 0.0
        else:
            return (h-h_k) * math.sqrt(1/2*g*((h+h_k)/(h*h_k)))

def f_k_tensor(h, h_s, h_k):
    valid = torch.relu(h)/torch.abs(h) # true wave condition 
    rarefaction = torch.relu(h_k-h_s)/(torch.abs(h_k-h_s)+1e-10) # 1, when h_s <= h_k meaning rarefaction or 0 for shock shock condition
    shock = torch.relu(h_s-h_k)/(torch.abs(h_s-h_k)+1e-10) # 0, when h_s <= h_k meaning rarefaction or 1 for shock shock condition
    rarefaction_output = rarefaction * valid * 2 * (torch.sqrt(torch.abs(h) * 9.8 + 1e-10) - torch.sqrt(h_k * 9.8))
    shock_output = shock * valid * (h - h_k) * torch.sqrt((1/2) * 9.8 * (torch.abs(h + h_k) / torch.abs(h * h_k)))
    #shock_output = torch.where(valid_shock, h, torch.full_like(h, 1e-10))
    return rarefaction_output + shock_output  # Add unsqueeze to ensure a 1-dimensional tensor is returned

# Calculate the function f_k, where k is the left or right side of the interface
def f_k_numpy(g, h, h_k):
    #rarefaction
    rarefaction_mask = h <= h_k 
    mask_2 = np.logical_or(h_k < 10e-8, h < 10e-8)
    f_k = np.zeros_like(h)
    f_k[rarefaction_mask & ~mask_2] = 2*(np.sqrt(h[rarefaction_mask & ~mask_2]*g)-np.sqrt(h_k[rarefaction_mask & ~mask_2]*g)) 
    f_k[~rarefaction_mask & ~mask_2] = (h[~rarefaction_mask & ~mask_2] - h_k[~rarefaction_mask & ~mask_2])*np.sqrt(1/2*g*((h[~rarefaction_mask & ~mask_2]+h_k[~rarefaction_mask & ~mask_2])/(h[~rarefaction_mask & ~mask_2]*h_k[~rarefaction_mask & ~mask_2]))) 
    f_k[mask_2] = 10e-9
    return f_k

# Calculate the derivative of f_k
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
        
# This function is in (Toro (10.23) - page 181)
def qk(h_s, h_k): 
    if (h_s > h_k):
        return math.sqrt(1/2*(h_s+h_k)*h_s/(h_k**2))
    else:
        return 1.0

# Get the speed of the dry/wet waves all four cases
def get_dry_speeds(u_l, a_l, u_r, a_r):
    return (u_r - 2*a_r, u_r + a_r, u_l + 2*a_l, u_l - a_l) # dry/wet front right, head right rarefaction, dry/wet front left, head left rarefaction
 
# Return the flux from the primitive variables
def flux_from_w(W, g):
    return np.array([W[0]*W[1], W[0]*(W[1]**2) + 0.5*g*(W[0]**2), W[0]*W[1]*W[2]])