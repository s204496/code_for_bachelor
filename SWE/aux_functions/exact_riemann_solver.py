import sys
import math
from aux_functions import f, wet_bed, sampler

def solve(out_file, bool_output, s_t_ratio, h_l, u_l, psi_l, h_r, u_r, psi_r, g, tolerance, iteration):
    
    #computing celerity on the left and right side
    a_l = math.sqrt(g*h_l)
    a_r = math.sqrt(g*h_r)

    # we check whether the depth posittivity condition is satisfied, you can see this condition in Toro - Shock-cap... - page 100
    dpc = 2*(a_l + a_r) >= (u_r - u_l)

    # Dry bed case
    if (not(dpc) or h_l <= 0 or h_r <= 0):
        if bool_output:
            out_file.write('Case: Dry bed\n')
        (s_sr, s_hr, s_sl, s_hl) = f.get_dry_speeds(h_l, u_l, a_l, h_r, u_r, a_r)
        (h_x, u_x, psi_x) = sampler.single_sample_dry(g, s_t_ratio, s_sr, s_hr, s_sl, s_hl, h_l, u_l, psi_l, a_l, h_r, u_r, psi_r, a_r)
        return (True, h_x, u_x, psi_x) # no star region means h_s = 0.0, u_s = 0.0, psi_s = 0.0, and a_s = 0.0
    else: # Wet bed case   
        if bool_output:
            out_file.write('Case: Wet bed\n')
        (h_s, u_s, a_s) = wet_bed.calculate(out_file, bool_output, g, tolerance, iteration, h_l, u_l, a_l, h_r, u_r, a_r)
        (h_x, u_x, psi_x) = sampler.single_sample_wet(g, s_t_ratio, h_l, u_l, psi_l, a_l, h_s, u_s, a_s, h_r, u_r, psi_r, a_r)
        return (False, h_x, u_x, psi_x)
    print("Should not get to this point in the program, something went wrong in the function exact_Riemann_solver")
    sys.exit(1)
    