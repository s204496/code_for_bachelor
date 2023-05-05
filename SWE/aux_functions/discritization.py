import math
import sys
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
        W[0][i] = U[0][i]
        if (U[0][i] <= 0):
            W[1][i] = 0.0
            W[2][i] = 0.0
            continue
        W[1][i] = U[1][i]/U[0][i]
        W[2][i] = U[2][i]/U[0][i]

def discritize_initial_values(x_len, cells, break_pos, h_l, u_l, psi_l, h_r, u_r, psi_r):
    U = [[],[],[]] #first is h, second is hu, third is h*psi, these are the conservative variables
    W = [[],[],[]] #first is h, second is u, third is psi, primitive variables
    for i in range(cells): # two more cells for the boundary conditions
        x_i = i*x_len/cells
        if(x_i < break_pos):
            U[0].append(h_l)
            U[1].append(h_l*u_l)
            U[2].append(h_l*psi_l)
            W[0].append(h_l)
            W[1].append(u_l)
            W[2].append(psi_l)
        else:
            U[0].append(h_r)
            U[1].append(h_r*u_r)
            U[2].append(h_r*psi_r)
            W[0].append(h_r)
            W[1].append(u_r)
            W[2].append(psi_r)
    # boundary conditions
    for j in range(3):
        U[j].insert(0,U[j][0])
        W[j].insert(0,W[j][0])
        U[j].append(U[j][cells])
        W[j].append(W[j][cells])
    return (U,W)

def evolve(U, fluxes, x_len, delta_t, cells):
    delta_x = x_len/cells
    for i in range(1,cells+1):
        for j in range(3):
            U[j][i] = U[j][i] - (delta_t/delta_x)*(fluxes[j][i] - fluxes[j][i-1])
    # boundary conditions
    for j in range(3):
        U[j][0] = U[j][1]
        U[j][cells+1] = U[j][cells]

def boundary_fluxes(bool_output, out_file, W, g, cells, solver, x_len, tolerance, iteration, CFL): 
    S_max = -1.0
    boundary_flux = [[],[],[]]
    for i in range(cells+1):
        dry_bool_c = False
        if solver == 0: #exact
            (dry_bool, (h_x, u_x, psi_x), (_ , _)) = exact_riemann_solver.solve(bool_output, out_file, 0.0, W[0][i], W[1][i], W[2][i], W[0][i+1], W[1][i+1], W[2][i+1], g, tolerance, iteration)
            boundary_flux[0].append(h_x*u_x)
            boundary_flux[1].append(h_x*u_x*u_x + 0.5*g*(h_x**2))
            boundary_flux[2].append(h_x*u_x*psi_x)
            dry_bool_c = dry_bool
        elif solver == 1: # HLLC 
            (dry_bool, fluxes, (_, _)) = HLLC_riemann.solve(W[0][i], W[1][i], W[2][i], W[0][i+1], W[1][i+1], W[2][i+1], g)
            boundary_flux[0].append(fluxes[0])
            boundary_flux[1].append(fluxes[1])
            boundary_flux[2].append(fluxes[2])
            dry_bool_c = dry_bool
        if not(dry_bool_c): #in the weet bed case
            S = abs(W[1][i]) + math.sqrt(g*W[0][i])
            if(S_max < S):
                S_max = S 
        else: #in the dry bed case
            a_l, a_r = 0.0, 0.0
            if (W[0][i] > 0): 
                a_l = math.sqrt(g*W[0][i])
            if (W[0][i+1] > 0):
                a_r = math.sqrt(g*W[0][i+1])
            (s_sr, s_hr, s_sl, s_hl) = f.get_dry_speeds(W[0][i], W[1][i], a_l, W[0][i+1], W[1][i+1], a_r)
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

def single_wave_interface_speeds(W_l, W_r, g, riemann_int):
    # If we call the returned array S. 
    # S[0] is the type of wave:
    # 0 = both are dry, all other values are 0 
    # 1 = dry bed, dry-left, wet-right S[1] is S_sr, S[2] is S_r
    # 2 = dry bed, wet-left, dry-right S[1] is S_l, S[2] is S_sl 
    # 3 = dry bed, dry midel, wet-left, wet-right S[1] is S_l, S[2] is S_sl, S[3] is S_sr, S[4] is S_r
    # 4 = wet bed, with rarefaction S[1] is S_l, S[2] S_s, S[3] is S_r
    # 5 = wet bed, without rarefaction S[1] is S_l, S[2] is S_s, S[3] is S_r
    dry, boundary_fluxes, h_u_s = None, None # Just for scoping reasons
    if (riemann_int == 0):
        (dry, boundary_fluxes, h_u_s) = exact_riemann_solver.solve(False, None, 0.0, W[0], W[1], W[2], W[0][i+1], W[1][i+1], W[2][i+1], g, 1e-6, 100)
    if (riemann_int == 1):
        (dry, boundary_fluxes, h_u_s) = HLLC_riemann.solve(False, None, 0.0, W[0][i], W[1][i], W[2][i], W[0][i+1], W[1][i+1], W[2][i+1], g, 1e-6, 100)
    if (dry_bed):
        if (W_l[0] <= 0 and W_r[0] <= 0): #S[0] = 0
            return [0, 0.0, 0.0, 0.0, 0.0]
        elif (W_l[0] <= 0): # S[0] = 1
            return [1, W_r[1]-2*math.sqrt(W_r[0]*g), W_r[1]+math.sqrt(W_r[0]*g), 0.0, 0.0]
        elif (W_l[0] <= 0): # S[0] = 2
            return [2, W_l[1]-math.sqrt(W_l[0]*g), W_l[1]+2*math.sqrt(W_l[0]*g), 0.0, 0.0]
        else: # S[0] = 3
            return [3, W_l[1]-math.sqrt(W_l[0]*g), W_l[1]+2*math.sqrt(W_l[0]*g), W_r[1]-2*math.sqrt(W_r[0]*g), W_r[1]+math.sqrt(W_r[0]*g)]
    else: 
        if (u_s > 0 and h_s <= h_l and not(np.sign(u_l - a_l) == np.sign(u_s-math.sqrt(h_s*g)))) or (u_s < 0 and h_s <= h_r and not(np.sign(u_r + a_r) == np.sign(u_s+math.sqrt(h_s*g)))): #S[0] = 4
            return [4, W_l[1]-math.sqrt(W_l[0]*g), u_s,  W_r[1]+math.sqrt(W_r[0]*g), 0] 
        else: # S[0] = 5
            return [5, W_l[1]-math.sqrt(W_l[0]*g), u_s,  W_r[1]+math.sqrt(W_r[0]*g), 0] 

def get_c_delta_flux(i, S, W, delta_t, delta_x, boundary_flux):
    # c has 4 values, if there is less the 4 wave interfaces, the last values are just 0. The only case, where there are 4 importent wave speeds is dry middel
    match S[0]:
        case 0:
            return (0, [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        case 1:
            delta_w_1 = np.array([0.0, 0.0, 0.0]) - np.array(w_from_u(boundary_flux[i+1])) # flux jump across wave 1 
            delta_flux_2 = np.array(boundary_flux[i+1]) - np.array(flux_from_W(W[i+1])) # flux jump across shear wave
            return (3, [delta_x/delta_t*S[i][1], delta_x/delta_t*S[i][2], delta_x/delta_t*S[i][2], 0.0, 0.0, 0.0], [delta_flux_1, delta_flux_2, delta_flux_2, 0.0 , 0.0, 0.0])
        case 2:
            delta_flux_1 = np.array(flux(W[i][0])) - np.array(boundary_flux[i+1]) # flux jump across wave 1 
            delta_flux_2 = np.array(np.array(boundary_flux[i+1])) - np.array([0.0,0.0,0.0]) # flux jump across shear wave
            return (3, [delta_x/delta_t*S[i][1], delta_x/delta_t*S[i][2], delta_x/delta_t*S[i][2], 0.0, 0.0, 0.0], [delta_flux_1, delta_flux_2, delta_flux_2, 0.0 , 0.0, 0.0])
        case 3:
            delta_flux_1 = np.array(flux(W[i][0])) - np.array(0.0, 0.0, 0.0) # flux jump across wave 1 
            delta_flux_2 = np.array(0.0, 0.0, 0.0) - np.array(flux(W[i+1][0])) # flux jump across wave 1 
            return (6, [delta_x/delta_t*S[1], delta_x/delta_t*S[1], delta_x/delta_t*S[1], delta_x/delta_t*S[3], delta_x/delta_t*S[3], delta_x/delta_t*S[3]], [delta_w_1, delta_w_1, delta_w_1, delta_w_2, delta_w_2, delta_w_2])
        case 4:
            delta_flux_1, delta_flux_2, delta_w_3 = None, None, None
            if S[2] < 0: # left sonic rarefaction
                delta_w_1 = np.array(flux(W[i][0])) - np.array(boundary_flux[i+1]) # flux jump across wave 1 
                delta_w_2 = np.array(boundary_flux[i+1]) - np.array([h_s, u_s, ])) # flux jump across wave 1
            delta_w_2 = np.array(0.0, 0.0, 0.0) - np.array(flux(W[i+1][0])) # flux jump across wave 1 
            return (3, [delta_x/delta_t*S[1], delta_x/delta_t*S[2], delta_x/delta_t*S[3], 0.0, 0.0, 0.0])
        case 5:
            return (3, [delta_x/delta_t*S[1], delta_x/delta_t*S[2], delta_x/delta_t*S[3], 0.0])

def flux_WAF_TVD(W, g, riemann_int, cells, delta_t, delta_x):
    S = [] 
    c = []
    for i in range(cells+1):
        S_i = single_wave_interface_speeds(W[:][i], W[:][i+1], g, riemann_int)
        S.append(single_wave_interface_speeds(W[:][i], W[:][i+1], g, riemann_int))        
    for i in range(cells+1):
        n_waves, c_k, r_k, delta_flux_k = get_c_r_delta_flux(i, S, delta_t, delta_x, boundary_flux)
    S = np.array(S)
    for wave_familiy in S:

    return [critical, c_k]
