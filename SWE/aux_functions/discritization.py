import math
import sys
import numpy as np
from aux_functions import f,exact_riemann_solver, HLLC_riemann

def flux_lax_friedrich(W, U, g, cells, dx, dt):
    fluxes_at_boundaries = [[],[],[]]
    fluxes_cells = [[],[],[]]
    # define u1-3 for scope
    for i in range(cells+2): # compute fluxes for cells
        u1 = W[0][i]
        if (u1 <= 0):#this is just and edge case, to avoid division by zero due to float inaccuracy, when number of cells is very large
            fluxes_cells[0].append(0.0)
            fluxes_cells[1].append(0.0)
            fluxes_cells[2].append(0.0)
            continue
        u2 = u1 * W[1][i]
        u3 = u1 * W[2][i]
        fluxes_cells[0].append(u2)
        fluxes_cells[1].append((u2**2)/u1 + 0.5*g*(u1**2))
        fluxes_cells[2].append((u2*u3)/u1)
    for i in range(cells+1): # compute fluex for boundaries with the fluxes for cells given above
        fluxes_at_boundaries[0].append(0.5*(fluxes_cells[0][i] + fluxes_cells[0][i+1]) + 0.5*(dx/dt)*(U[0][i]-U[0][i+1]))
        fluxes_at_boundaries[1].append(0.5*(fluxes_cells[1][i] + fluxes_cells[1][i+1]) + 0.5*(dx/dt)*(U[1][i]-U[1][i+1]))
        fluxes_at_boundaries[2].append(0.5*(fluxes_cells[2][i] + fluxes_cells[2][i+1]) + 0.5*(dx/dt)*(U[2][i]-U[2][i+1]))
    return fluxes_at_boundaries 

def W_from_U(U, W, cells): # this method overwrites the content of W based on U
    for i in range(cells+2):
        W[i][0] = U[i][0]
        if (U[i][0] <= 0):
            W[i][1] = 0.0
            W[i][2] = 0.0
            continue
        W[i][1] = U[i][1]/U[i][0]
        W[i][2] = U[i][2]/U[i][0]

def discritize_initial_values(x_len, cells, break_pos, h_l, u_l, psi_l, h_r, u_r, psi_r):
    U = [] #first is h, second is hu, third is h*psi, these are the conservative variables
    W = [] #first is h, second is u, third is psi, primitive variables
    for i in range(cells): # two more cells for the boundary conditions
        x_i = i*x_len/cells
        if(x_i < break_pos):
            U.append([h_l, h_l*u_l, h_l*psi_l])
            W.append([h_l, u_l, psi_l])
        else:
            U.append([h_r, h_r*u_r, h_r*psi_r])
            W.append([h_r, u_r, psi_r])
    # boundary conditions
    U.insert(0,U[0])
    U.append(U[cells])
    W.insert(0,W[0])
    W.append(W[cells])
    return (U,W)

def evolve(U, fluxes, x_len, delta_t, cells):
    delta_x = x_len/cells
    for i in range(1,cells+1):
        for j in range(3):
            U[i][j] = U[i][j] - (delta_t/delta_x)*(fluxes[i][j] - fluxes[i-1][j])
    # boundary conditions
    for j in range(3):
        U[0][j] = U[1][j]
        U[cells+1][j] = U[cells][j]

def fluxes_at_boundary(bool_output, out_file, W, g, cells, solver, x_len, tolerance, iteration, CFL): 
    S_max = -1.0
    boundary_flux = []
    for i in range(cells+1):
        dry_bool_c = False
        if solver == 0: #exact
            (dry_bool, (_, _, _), fluxes, (_ , _)) = exact_riemann_solver.solve(bool_output, out_file, 0.0, W[i][0], W[i][1], W[i][2], W[i+1][0], W[i+1][1], W[i+1][2], g, tolerance, iteration)
            boundary_flux.append(fluxes)
            dry_bool_c = dry_bool
        elif solver == 1: # HLLC 
            (dry_bool, (_, _, _), fluxes, (_ , _)) = HLLC_riemann.solve(W[i][0], W[i][1], W[i][2], W[i+1][0], W[i+1][1], W[i+1][2], g)
            boundary_flux.append(fluxes)
            dry_bool_c = dry_bool
        if not(dry_bool_c): #in the weet bed case
            S = abs(W[i][1]) + math.sqrt(g*W[i][0])
            if(S_max < S):
                S_max = S 
        else: #in the dry bed case
            a_l, a_r = 0.0, 0.0
            if (W[i][0] > 0): 
                a_l = math.sqrt(g*W[i][0])
            if (W[i+1][0] > 0):
                a_r = math.sqrt(g*W[i+1][0])
            (s_sr, s_hr, s_sl, s_hl) = f.get_dry_speeds(W[i][0], W[i][1], a_l, W[i+1][0], W[i+1][1], a_r)
            max_dry_speed = max(abs(s_sr), abs(s_hr), abs(s_sl), abs(s_hl))
            if(S_max < max_dry_speed):
                S_max = max_dry_speed 
    delta_x = x_len/cells
    delta_t = CFL*delta_x/S_max
    return (delta_t, boundary_flux)

def center_dt_wet(W, cells, g, cfl, dx):
    S_max = -1.0
    for i in range(1,cells+1):
        S = abs(W[1][i]) + math.sqrt(g*W[0][i])
        if(S_max < S):
            S_max = S
    dt = cfl*dx/S_max
    return dt

def interface_wave_speeds(W_l, W_r, g, riemann_int, tolenrance, iterations):
    # This function returns (S, h_s, u_s, boundary_w), where u_s, h_s is the values in the star region, and boundary_w is the primite values at the interface
    # The first return parameter is S, that contains information about the wave speed, further the first entry contains information about the types of waves. 
    # S[0] is the type of wave:
    # 0 = same level on both sides, means no waves, from riemann problem
    # 1 = both are dry, all other values are 0 
    # 2 = dry bed, dry-left, wet-right S[1] is S_sr, S[2] is S_r
    # 3 = dry bed, wet-left, dry-right S[1] is S_l, S[2] is S_sl 
    # 4 = dry bed, dry midel, wet-left, wet-right S[1] is S_l, S[2] is S_r
    # 5 = wet bed, with critical left rarefaction S[1] is S_hl, S[2] S_s, S[3] is S_r
    # 6 = wet bed, with critical right rarefaction S[1] is S_l, S[2] S_s, S[3] is S_hr
    # 7 = wet bed, without rarefaction S[1] is S_l, S[2] is S_s, S[3] is S_r
    if (W_l == W_r ): #S[0] = 0
        return ([0, 0.0, 0.0, 0.0], W_l[0], W_l[1], W_l)
    dry, _, h_u_s = None, None, None # Just for scoping reasons
    if (riemann_int == 0):
        (dry, (h_x, u_x, psi_x), _, h_u_s) = exact_riemann_solver.solve(False, None, 0.0, W_l[0], W_l[1], W_l[2], W_r[0], W_r[1], W_r[2], g, tolenrance, iterations)
    if (riemann_int == 1):
        (dry, (h_x, u_x, psi_x), _, h_u_s) = HLLC_riemann.solve(W_l[0], W_l[1], W_l[2], W_r[0], W_r[1], W_r[2], g)
    if (dry):
        if (W_l[0] <= 0 and W_r[0] <= 0): #S[0] = 1
            return ([1, 0.0, 0.0, 0.0], h_u_s[0], h_u_s[1], [h_x, u_x, psi_x])
        elif (W_l[0] <= 0): # S[0] = 2
            return ([2, W_r[1]-2*math.sqrt(W_r[0]*g), W_r[1]+math.sqrt(W_r[0]*g), 0.0], h_u_s[0], h_u_s[1], [h_x, u_x, psi_x])
        elif (W_r[0] <= 0): # S[0] = 3
            return ([3, W_l[1]-math.sqrt(W_l[0]*g), W_l[1]+2*math.sqrt(W_l[0]*g), 0.0], h_u_s[0], h_u_s[1], [h_x, u_x, psi_x])
        else: # S[0] = 4
            return ([4, W_l[1]-math.sqrt(W_l[0]*g), W_r[1]+math.sqrt(W_r[0]*g), 0.0], 0.0, 0.0, [0.0, 0.0, 0.0])
    else: 
        if (h_u_s[1] > 0 and h_u_s[0] <= W_l[0] and not(np.sign(W_l[1] - math.sqrt(g*W_l[0])) == np.sign(h_u_s[1]-math.sqrt(h_u_s[0]*g)))): #S[0] = 5
            return ([5, W_l[1]-math.sqrt(W_l[0]*g), h_u_s[1],  W_r[1]+math.sqrt(W_r[0]*g)*f.qk(h_s, h_r), 0], h_u_s[0], h_u_s[1], [h_x, u_x, psi_x])
        elif(h_u_s[1] < 0 and h_u_s[0] <= W_r[0] and not(np.sign(W_r[1] + math.sqrt(W_r[0]*g)) == np.sign(h_u_s[1]+math.sqrt(h_u_s[0]*g)))): #S[0] = 6
            return ([6, W_l[1]-math.sqrt(W_l[0]*g)*f.qk(h_s, h_l), h_u_s[1],  W_r[1]+math.sqrt(W_r[0]*g), 0], h_u_s[0], h_u_s[1], [h_x, u_x, psi_x])
        else: # S[0] = 7
            return ([7, W_l[1]-math.sqrt(W_l[0]*g), h_u_s[1],  W_r[1]+math.sqrt(W_r[0]*g), 0], h_u_s[0], h_u_s[1], [h_x, u_x, psi_x])

####!!!! came to here, fix the fact that S[0] has range values 0-7 boundaries included, but not all are handled here
def get_c_dw_dflux(S, w_l, w_r, delta_t, delta_x, boundary_flux, boundary_w, h_s, u_s):
    # return (number of waves, list_of_c_for_each_wave, list_of_jump_in_w, list_of_jump_in_flux) 
    match S[0]:
        case 0:
            return (0, w_l, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]) 
        case 1:
            return (0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]) 
        case 2:
            delta_w_1 = np.array([0.0, 0.0, 0.0]) - np.array(boundary_w) # w jump across wave 1 
            delta_w_2 = np.array(boundary_w) - np.array(W_r) # w jump across shear wave
            delta_flux_1 = np.array([0.0, 0.0, 0.0]) - np.array(boundary_flux) # flux jump across wave 1 
            delta_flux_2 = np.array(boundary_flux) - np.array(f.flux_from_w(w_r[0], w_r[1], w_r[2], g)) # flux jump across wave 1 
            return (2, [delta_x/delta_t*S[i][1], delta_x/delta_t*S[i][2], delta_x/delta_t*S[i][2]], [delta_w_1, delta_w_2, delta_w_2], [delta_flux_1, delta_flux_2, delta_flux_2])
        case 3:
            delta_w_1 = np.array(w_l) - np.array(boundary_w) # w jump across wave 1 
            delta_w_2 = np.array(boundary_w) - np.array([0.0,0.0,0.0]) # w jump across shear wave
            delta_flux_1 = np.array(f.flux_from_w(w_l[0], w_l[1], w_l[2], g)) - np.array(boundary_flux) # flux jump across wave 1 
            delta_flux_2 = np.array(boundary_flux) - np.array([0.0,0.0,0.0]) # flux jump across shear wave
            return (2, [delta_x/delta_t*S[i][1], delta_x/delta_t*S[i][2], delta_x/delta_t*S[i][2]], [delta_w_1, delta_w_2, delta_w_2])
        case 4:
            delta_w_1 = np.array(w_l) - np.array(0.0, 0.0, 0.0) # w jump across wave 1 
            delta_w_2 = np.array(0.0, 0.0, 0.0) - np.array(w_r) # w jump across wave 2 
            delta_flux_1 = np.array(f.flux_from_w(w_l[0], w_l[1], w_l[2], g)) - np.array(0.0, 0.0, 0.0) # flux jump across wave 1 
            delta_flux_2 = np.array(0.0, 0.0, 0.0) - np.array(w_r[0], w_r[1], w_r[2], g) # flux jump across wave 2 
            return (2, [delta_x/delta_t*S[1], delta_x/delta_t*S[3], delta_x/delta_t*S[3]], [delta_w_1, delta_w_2, delta_w_2], [delta_flux_1, delta_flux_2, delta_flux_2])
        case 5:
            delta_w_1, delta_w_2, delta_w_3 = None, None, None
            if S[2] > 0: # left sonic rarefaction
                delta_w_1 = np.array(w_l) - np.array(boundary_w) # w jump across wave 1 
                delta_w_2 = np.array(boundary_w) - np.array(h_s, u_s, w_r[2]) # w jump across wave 2
                delta_w_3 = np.array(h_s, u_s, w_r[2]) - np.array(w_r) # w jump across wave 3
                delta_flux_1 = np.array(f.flux_from_w(w_l[0], w_l[1], w_l[2], g)) - np.array(boundary_flux) # flux jump across wave 1 
                delta_flux_2 = np.array(boundary_flux) - np.array(f.flux_from_w(h_s, u_s, w_r[2], g)) # flux jump across wave 2 
                delta_flux_3 = np.array(f.flux_from_w(h_s, u_s, w_r[2], g)) - np.array(f.flux_from_w(w_r[0], w_r[1], w_r[2], g)) # flux jump across wave 3 
                return (3, [delta_x/delta_t*S[1], delta_x/delta_t*S[2], delta_x/delta_t*S[3]], [delta_w_1, delta_w_2, delta_w_3], [delta_flux_1, delta_flux_2, delta_flux_3])
            else: # right sonic rarefaction
                delta_w_1 = np.array(w_l) - np.array(h_s, u_s, w_l[2]) # w jump across wave 1 
                delta_w_2 = np.array(h_s, u_s, w_l[2]) - np.array(boundary_w) # w jump across wave 2
                delta_w_3 = np.array(np.array(boundary_w)) - np.array(w_r) # w jump across wave 3
                delta_flux_1 = np.array(f.flux_from_w(w_l[0], w_l[1], w_l[2], g)) - np.array(f.flux_from_w(h_s, u_s, w_l[2], g)) # flux jump across wave 1 
                delta_flux_2 = np.array(f.flux_from_w(h_s, u_s, w_l[2], g)) - boundary_flux # flux jump across wave 2 
                delta_flux_3 = boundary_flux - np.array(f.flux_from_w(w_r[0], w_r[1], w_r[2], g)) # flux jump across wave 3 
                return (3, [delta_x/delta_t*S[1], delta_x/delta_t*S[2], delta_x/delta_t*S[3]], [delta_w_1, delta_w_2, delta_w_3], [delta_flux_1, delta_flux_2, delta_flux_3])
        case 6:
            delta_w_1 = np.array(w_l) - np.array(h_s, u_s, w_l[2]) # w jump across wave 1 
            delta_w_2 = np.array(h_s, u_s, w_l[2]) - np.array(h_s, u_s, w_r[2]) # w jump across wave 2
            delta_w_3 = np.array(h_s, u_s, w_l[2]) - np.array(w_r) # w jump across wave 3
            delta_flux_1 = np.array(f.flux_from_w(w_l[0], w_l[1], w_l[2], g)) - np.array(f.flux_from_w(h_s, u_s, w_l[2], g)) # flux jump across wave 1 
            delta_flux_2 = np.array(f.flux_from_w(h_s, u_s, w_l[2], g)) - np.array(f.flux_from_w(h_s, u_s, w_r[2], g)) # flux jump across wave 2 
            delta_flux_3 = np.array(f.flux_from_w(h_s, u_s, w_r[2], g)) - np.array(f.flux_from_w(w_r[0], w_r[1], w_r[2], g)) # flux jump across wave 3 
            return (3, [delta_x/delta_t*S[1], delta_x/delta_t*S[2], delta_x/delta_t*S[3]], [delta_w_1, delta_w_2, delta_w_3], [delta_flux_1, delta_flux_2, delta_flux_3])

def super_bee_limiter(w_jumps_l, w_jumps_m, w_jumps_r, c):
    r = None # just for scope
    if c > 0:
        r = w_jumps_l/w_jumps_m
    else:
        r = w_jumps_r/w_jumps_m
    if r <= 0:
        return 1
    elif r > 0 and r <= 0.5:
        return 1-2*(1-np.abs(c))*r 
    elif r > 0.5 and r <= 1:
        return np.abs(c)
    elif r > 1 and r <= 2:
        return 1-(1-np.abs(c))*r 
    else:
        return 2*np.abs(c)-1 
        

def flux_WAF_TVD(W, g, riemann_int, cells, delta_t, delta_x, boundary_flux, tolenrance, iterations):
    S_list = [] 
    c = []
    n_waves_list = []
    boundary_w_list = []
    h_u_s = []
    dFlux_list = []
    w_jumps = []
    waf_flux = []
    for i in range(cells+1):
        S, u_s, h_s, boundary_w = interface_wave_speeds(W[:][i], W[:][i+1], g, riemann_int, tolenrance, iterations)
        S_list.append(S)
        h_u_s.append([h_s, u_s])
        boundary_w_list.append(boundary_w)
    for i in range(cells+1):
        waves, c_k, delta_w, delta_flux_k = get_c_dw_dflux(S_list[i], W[:][i], W[:][i+1], delta_t, delta_x, boundary_flux[i], boundary_w_list[i], h_u_s[i][0], h_u_s[i][1])
        n_waves_list.append(waves)
        c.append(c_k)
        w_jumps.append(delta_w)
        dFlux_list.append(delta_flux_k)
    for i in range(cells+1):
        waf_flux_without_sum = 0.5*(np.array(f.flux_from_w(W[i][0], W[i][1], W[i][2], g)) + np.array(f.flux_from_w(W[i][0], W[i][1], W[i][2], g)))
        waf_sum = 0
        for j in range(n_waves_list[i]):
            waf_sum = waf_sum + np.sign(c[i][j])*super_bee_limiter(w_jumps[i-1][j], w_jumps[i][j], w_jumps[i+1][j], c[i][j])*dFlux_list[i][j]
        waf_flux.append(waf_flux_without_sum - waf_sum)
    return waf_flux 
