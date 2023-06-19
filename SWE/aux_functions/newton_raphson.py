import sys
import math
from tabulate import tabulate
from aux_functions import f

"""
This function is the Newton-Raphson method, it takes in the initial guess, the function, the derivative of the function, the tolerance and the maximum number of iterations.
The purpose of the function is to find height h_s of the water in between two waves, under the assumption, that the depth positivity condition is satiesfied. 
We are trying to solve the equation f(h)=f_l(h,h_l) + f_r(h,h_r) + u_r - u_l = 0 (Toro (6.4) - page 101)
we do this by iteratively solving the equation h_n+1 = h_n - f(h_n)/f'(h_n)
"""
def newton_rapson_iter(g, h_0, h_l, u_l, a_l, h_r, u_r, a_r, tolerance, max_iterations):
    h_spre, h_s = h_0, h_0
    iter = 0
    for i in range(max_iterations):
        f_l = f.f_k(g, h_s, h_l)
        f_r = f.f_k(g, h_s, h_r)
        f_ld =f.fkd(g, h_s, h_l, a_l)
        f_rd =f.fkd(g, h_s, h_r, a_r)
        if not(f_ld <= 0 or f_rd <= 0): # need this to deal with edge case introduced by large cell number in dry bed case
            h_s = h_s - (f_l+f_r+u_r-u_l)/(f_ld+f_rd) # this is the newton raphson step
        h_delta = abs(h_spre-h_s)/(0.5*(h_spre + h_s)) # this is a relative change (Toro (5.25) - page 102)
        iter = iter+1
        if h_delta < tolerance:
            break
        if(h_s < 0):
           h_s = tolerance 
        h_spre = h_s
        if(i == (max_iterations -1)):
            return h_s
    return h_s

"""
The purpose of this function is to find an initial guess for the Newton-Raphson method. 
If both the right and left going waves are rarefaction waves, then the exact solution is (1/g)*((1/2*(a_l+a_r)-(1/4)*(u_r-u_l))**2).
If this is not the case we use it as the initial guess 
"""
def initial_guess(g, h_l, u_l, a_l, h_r, u_r, a_r):
    h_min = min(h_l, h_r)
    # if f(h_min) >= 0, then we have two rarefaction waves.
    h_s = (1/g)*((1/2*(a_l+a_r)-(1/4)*(u_r-u_l))**2)
    if f.f(g, h_min, h_l, u_l, h_r, u_r) >= 0:
        return (1/g)*((1/2*(a_l+a_r)-(1/4)*(u_r-u_l))**2)
    #we are not in the two rarefaction case.
    else:
        # we use the two shock approximation to h_s, by the formular (Toro (10.19) - page 179)
        g_l = math.sqrt(1/2*g*((h_s+h_l)/(h_s*h_l)))  # g_l and g_r are the values calculated in (Toro (5.14) - page 99)
        g_r = math.sqrt(1/2*g*((h_s+h_r)/(h_s*h_r)))
        return (g_l*h_l+g_r*h_r+u_l-u_r)/(g_l+g_r)