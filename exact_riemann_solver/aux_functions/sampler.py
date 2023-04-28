"""The purpose of this class is to sample the solution at a given time t_end. 
For both the dry bed case and the wet bed case. 
t is a fixed value, and we want to sample the solution over x, which is discretized into a number of points
The number of points is given by the variable cells."""

import math, sys
import f

def single_sample_wet(s, h_l, h_s, h_r, u_l, u_s, u_r, a_l, a_s, a_r, psi_l, psi_r):
    if (s <= u_s): # to the left of the shear wave
        if (h_s > h_l): # the left wave is a shock wave
            q_l = math.sqrt(0.5*((h_s + h_l)*h_s)/(h_l**2))
            s_l = u_l - a_l*q_l # the left shock speed 
            if (s <= s_l): # to the left of the left shock
                return (h_l, u_l, psi_l)
            else: # to the right of the left shock
                return (h_s, u_s, psi_l)
        else: # the left wave is a rarefaction wave
            s_hl = u_l - a_l # the speed of the head of rarefaction wave
            s_tl = u_s - a_s # the speed of the tail of rarefaction wave
            if(s <= s_hl): # to the left of the rarefaction
                return (h_l, u_l, psi_l)
            elif(s <= s_tl): # inside rarefaction wave 
                u_x = (u_l+2*a_l+2*s)/3
                a_x = (u_l+2*a_l-s)/3
                h_x = (a_x**2)/g
                return (h_x, u_x, psi_l)
            else: # to the right of the rarefaction
                return (h_s, u_s, psi_l)
    else: # to the right of the shear wave
        if (h_s > h_r): # the right wave is a shock wave
            q_r = math.sqrt(0.5*((h_s + h_r)*h_s)/(h_r**2))
            s_r = u_r + a_r*q_r # the right shock speed 
            if (s < s_r): # to the left of the right shock
                return (h_s, u_s, psi_r)
            else: # to the right of the right shock
                return (h_r, u_r, psi_r)
        else: # the right wave is a rarefaction wave
            s_hr = u_r + a_r # the speed of the head of rarefaction wave
            s_tr = u_s + a_s # the speed of the tail of rarefaction wave
            if(s <= s_tr): # to the left of the rarefaction
                return (h_s, u_s, psi_r)
            elif(s <= s_hr): # inside rarefaction wave 
                u_x = (u_r-2*a_r+2*s)/3
                a_x = (-u_r+2*a_r+s)/3
                h_x = (a_x**2)/g
                return (h_x, u_x, psi_r)
            else: # to the right of the rarefaction
                return (h_r, u_r, psi_r)
    print("Something went completely wrong in the single_sample_wet function")
    sys.exit(1)

# The wet bed case
def sample_domain_wet(out_file, to_output, x_len, break_pos, t_end, cells, g, h_l, h_s, h_r, u_l, u_s, u_r, a_l, a_s, a_r):
    if to_output:
        out_file.write("Sampling the solution at t = " + str(t_end) + " with " + str(cells) + " cells:\n\n")
    sol_data = []
    for i in range(cells+1):
        x_i = i*(x_len/cells)-break_pos # moving the break position to x=0
        s = x_i/t_end # the similarity variable
        (h_x, u_x, psi_x) = single_sample_wet(s, h_l, h_s, h_r, u_l, u_s, u_r, a_l, a_s, a_r, psi_l, psi_r)
        sol_data.append((i, x_i+break_pos, h_x, u_x, psi_x))
        if to_output:
            out_file.write(str((i, x_i+break_pos, h_x, u_x, psi_x)) + " ")
    return sol_data

def single_sample_wet(s, s_sr, s_hr, s_sl, s_hl, h_l, h_r, u_l, u_r, a_l, a_r, psi_l, psi_r):
    if(h_l <= 0): # the left is dry
        if (s <= s_sr): # to the left of the dry/wet front
            return (h_l, u_l, psi_l) # all these values should be 0
        elif(s <= s_hr): # inside the rarefaction wave
            u_x = (u_r-2*a_r+2*s)/3
            a_x = (-u_r+2*a_r+s)/3
            h_x = (a_x**2)/g
            return (h_x, u_x, psi_r)
        else: # to the right of the rarefaction
            return (h_r, u_r, psi_r)
    elif(h_r <= 0): # the right is dry
        if (s <= s_hl): # to the left of the rarefaction
            return (h_l, u_l, psi_l)
        elif(s <= s_sl): # inside the rarefaction wave
            u_x = (u_l+2*a_l+2*s)/3
            a_x = (u_l+2*a_l-s)/3
            h_x = (a_x**2)/g
            return (h_x, u_x, psi_l)
        else: # to the right of the dry/wet front
            return (h_r, u_r, psi_r)
    else: # the dry bed is created in the middel 
        if (s <= s_hl): # to the left of the rarefaction
            return (h_l, u_l, psi_l)
        elif (s <= s_sl): # in the left rarefaction
            u_x = (u_l+2*a_l+2*s)/3
            a_x = (u_l+2*a_l-s)/3
            h_x = (a_x**2)/g
            return (h_x, u_x, psi_l)
        elif (s <= s_sr): # in the dry region
            return (0.0, 0.0, 0.0)
        elif (s <= s_hr): # in the right rarefaction
            u_x = (u_r-2*a_r+2*s)/3
            a_x = (-u_r+2*a_r+s)/3
            h_x = (a_x**2)/g
            return (h_x, u_x, psi_r)
        else: # to the right of the rarefaction
            return (h_r, u_r, psi_r)
    print("something went completely wrong in the single_sample_wet function")

# The dry bed case
def sample_domain_dry(out_file, to_output, x_len, break_pos, t_end, cells, g, h_l, h_r, u_l, u_r, a_l, a_r):
    if to_output:
        out_file.write("Sampling the solution at t = " + str(t_end) + " with " + str(cells) + " cells:\n\n")
    sol_data = []
    for i in range(cells+1):
        x_i = i*(x_len/cells)-break_pos # moving the break position to x=0
        s = x_i/t_end # the similarity variable
        
        (s_sr, s_hr, s_sl, s_hl) = f.get_dry_speeds(h_l, h_r, u_l, u_r, a_l, a_r)
        (h_x, u_x, psi_x) = single_sample_dry(s, s_sr, s_hr, s_sl, s_hl, h_l, h_r, u_l, u_r, a_l, a_r, psi_l, psi_r)

        sol_data.append((i, x_i+break_pos, h_x, u_x, psi_x))
        if to_output:
            out_file.write(str((i, x_i+break_pos, h_x, u_x, psi_x)) + " ")
    return sol_data
