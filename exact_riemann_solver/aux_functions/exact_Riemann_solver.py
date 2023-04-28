import wet_bed, sampler

def exact_Riemann_solver(s_t_ratio, h_l, h_r, u_l, u_r, psi_l, psi_r, g, tolerance, iteration):
    
    #computing celerity on the left and right side
    a_l = math.sqrt(g*h_l)
    a_r = math.sqrt(g*h_r)

    # we check whether the depth posittivity condition is satisfied, you can see this condition in Toro - Shock-cap... - page 100
    dpc = 2*(a_l + a_r) >= (u_r - u_l)

    # Dry bed case
    if (not(dpc) or h_l <= 0 or h_r <= 0):
        (h_x, u_x, psi_x) = sampler.single_sample_dry(s_t_ratio, h_l, h_r, u_l, u_r, psi_l, psi_r)
        return (True, h_l, u_l, a_l, psi_l)
    else: # Wet bed case   
        (h_s, u_s, a_s) = wet_bed.calculate(out_file, g, tolerance, iteration, h_l, h_r, u_l, u_r, a_l, a_r)

    
    print("not implemented yet")
    dry_bool = False
    h_s = 1
    u_s = 1
    a_s = 1
    psi_s = 1
    return (dry_bool, h_s, u_s, a_s, psi_s)