import math
from aux_functions import newton_raphson, f

""" 
Purpose of this function is to solve the Riemann problem, to the given precision for the wet bed case,
The method calculates h_s, u_s and a_s. 
First it calculates an initial guess and then uses the iterative method newton_raphson, to find h_s.
u_s and a_s can be calculated in closed form, once h_s is known. 
"""

def calculate(g, tolerance, iterations, h_l, u_l, a_l, h_r, u_r, a_r):
    h_0 = newton_raphson.initial_guess(g, h_l, u_l, a_l, h_r, u_r, a_r)
    h_s = newton_raphson.newton_rapson_iter(g, h_0, h_l, u_l, a_l, h_r, u_r, a_r, tolerance, iterations)
    if (h_s < 0):
        h_s = 0.0001
    a_s = math.sqrt(g*h_s)
    f_l = f.f_k(g, h_s, h_l)
    f_r = f.f_k(g, h_s, h_r)
    u_s = 0.5*(u_l+u_r)+0.5*(f_r-f_l) # using the closed form given in (5.8) - Toro - Shock-cap... p.97
    return (h_s, u_s, a_s)