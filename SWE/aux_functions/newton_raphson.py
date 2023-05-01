import sys
import math
from tabulate import tabulate
from aux_functions import f

"""
This function is the Newton-Raphson method, it takes in the initial guess, the function, the derivative of the function, the tolerance and the maximum number of iterations.
The purpose of the function is to find height (h) of the water in between two waves, this is only when the depth positivity condition is satiesfied. 
We are trying to solve the equation f(h)=f_l(h,h_l) + f_r(h,h_r) + u_r - u_l = 0, which one can see on page 101 - Toro - Shock-cap...
we do this by iteratively solving the equation h_n+1 = h_n - f(h_n)/f'(h_n)
"""
def newton_rapson_iter(bool_out, out_file, g, h_0, h_l, u_l, a_l, h_r, u_r, a_r, tolerance, max_iterations):
    h_spre, h_s = h_0, h_0
    iter = 0
    table = None
    if bool_out:
        table = [['Iterations:', 'h_s', '\u0394h']]
        table.append([iter, h_s , 0])
    for i in range(max_iterations):
        f_l = f.f_k(g, h_s, h_l)
        f_r = f.f_k(g, h_s, h_r)
        f_ld =f.fkd(g, h_s, h_l, a_l)
        f_rd =f.fkd(g, h_s, h_r, a_r)
        h_s = h_s - (f_l+f_r+u_r-u_l)/(f_ld+f_rd) # this si the newton raphson step
        h_delta = abs(h_spre-h_s)/(0.5*(h_spre + h_s))#this is a relative change defined by (5.25) in Toro - Shock-cap...
        iter = iter+1
        if h_delta < tolerance:
            if bool_out:
                table.append([iter, h_s , h_delta])
            break
        if(h_s < 0):
           h_s = tolerance 
        h_spre = h_s
        if bool_out:
            table.append([iter, h_s , h_delta])
        if(i == (max_iterations -1)):
            if bool_out:
                out_file.write(tabulate(table))
                out_file.write('\nStopped Newton-Raphson due to max iterations')
            return h_s
    if bool_out:
        out_file.write(tabulate(table) + '\n')
        out_file.write("Number of iterations: " + str(iter) + '\n')
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
        # this is just to check that the definition given in book toro (5.3) and (5.4) match up with the other condition 5.18
        if not(0.0000001 >= h_s - h_min): 
            print("Error, Mark A: The condition given in 5.18 and (5.3-5.4) does not match up, this should not happen")
            sys.exit(1)
        return (1/g)*((1/2*(a_l+a_r)-(1/4)*(u_r-u_l))**2)
    #we are not in the two rarefaction case.
    else:
        # we use the two shock approximation to h_s, by the formular given in Toro - Shock-cap... p 179 (10.19)
        g_l = math.sqrt(1/2*g*((h_s+h_l)/(h_s*h_l)))  # g_l and g_r are the values calculated in Toro - p.99 - (5.14)
        g_r = math.sqrt(1/2*g*((h_s+h_r)/(h_s*h_r)))
        return (g_l*h_l+g_r*h_r+u_l-u_r)/(g_l+g_r)