import math
import sys
import numpy as np
from aux_functions import f,exact_riemann_solver, hllc_riemann
from data_driven.aux_function import riemann_aux, lax_godunov_flux_aux

# Computes the Lax-Friedrichs fluxes at the cell boundaries (see. Toro (9.29) - page 163)
def flux_lax(W, U, dx, cells, g, dt):
    fluxes_cells = np.empty([cells+2, 3])
    for i in range(cells+2):
        # If statement is just an edge case, to avoid division by zero due to float inaccuracy, happens when number of cells is very large
        if (W[i,0] <= 0): 
            fluxes_cells[i] = np.array([0.0, 0.0, 0.0])
        else:
            fluxes_cells[i] = f.flux_from_w(W[i], g)
    lax_flux_boundaries = np.array([(0.5*(fluxes_cells[i] + fluxes_cells[i+1]) + 0.5*(dx/dt)*(U[i]-U[i+1])) for i in range(cells+1)])
    return lax_flux_boundaries 

# Computes the Lax-Friedrichs fluxes with data driven method
def flux_lax_data_driven(W, U, dx, g, dt, model, data_driven_model):
    fluxes = lax_godunov_flux_aux.compute(model, W[:-1,:], W[1:,:], dx, dt, data_driven_model) 
    return fluxes 

# Computes the dt for the Lax-friedrichs scheme, not depend on Riemann solvers
def center_dt(W, dx, g, cfl):
    S_max = np.max(np.abs(W[:,1]) + np.sqrt(g*W[:,0]))
    return cfl*dx/S_max

# This method overwrites the content of W based on U
def W_from_U(U, W): 
    mask = U[:,0] > 0.0000000
    W[mask,0] = U[mask,0]
    W[mask,1:3] = U[mask,1:3]/U[mask,0].reshape(-1,1)
    W[~mask,:] = 0.0

# This method overwrites the content of U based on W
def U_from_W(U, W): 
    mask = W[:,0] > 0.0000000
    U[mask,0] = W[mask,0]
    U[mask,1] = W[mask,1]*W[mask,0]
    U[mask,2] = W[mask,2]*W[mask,0]
    W[~mask,:] = 0.0

# Computes the initial values of U and W, for 1D dame break propblem, U is conservative variables, W is primitive variables
def discretize_initial_values(dx, cells, break_pos, W_l, W_r):
    U, W = np.empty([cells+2, 3]), np.empty([cells+2, 3])
    for i in range(cells): 
        x_i = (i+0.5)*dx
        if(x_i < break_pos):
            U[i+1] = np.array([W_l[0], W_l[0]*W_l[1], W_l[0]*W_l[2]])
            W[i+1] = np.array([W_l[0], W_l[1], W_l[2]])
        else:
            U[i+1] = np.array([W_r[0], W_r[0]*W_r[1], W_r[0]*W_r[2]])
            W[i+1] = np.array([W_r[0], W_r[1], W_r[2]])
    # boundary conditions
    U[0], W[0], U[cells+1], W[cells+1] = U[1], W[1], U[cells], W[cells]
    return (U,W)

# Formular (Toro (8.8) - page 143)
def evolve(U, fluxes, delta_x, delta_t, cells):
    U[1:-1,:] = U[1:-1,:] - (delta_t/delta_x)*(fluxes[1:,:] - fluxes[0:-1,:])
    U[0,:], U[cells+1,:] = U[1,:], U[cells,:] 

# Computes the fluxes at the cell boundaries and dt based on (9.19 & 9.21 page 157-158)
def flux_at_boundaries(W, g, cells, solver, dx, tolerance, iteration, CFL): 
    S_max, boundary_flux = -1.0, np.empty([cells+1, 3])
    for i in range(cells+1):
        dry_bool_c = False
        if solver == 0: #exact
            (dry_bool_c, (_, _, _), boundary_flux[i], (_ , _)) = exact_riemann_solver.solve(0.0, W[i,:], W[i+1,:], g, tolerance, iteration)
        elif solver == 1: # HLLC 
            (dry_bool_c, (_, _, _), boundary_flux[i], (_ , _)) = hllc_riemann.solve(W[i,:], W[i+1,:], g)
        if not(dry_bool_c): # Weet bed case
            S = abs(W[i,1].item()) + math.sqrt(g*W[i,0].item())
            if(S_max < S):
                S_max = S 
        else: # Dry bed case
            a_l, a_r = 0.0, 0.0
            if (W[i,0] > 0): 
                a_l = math.sqrt(g*W[i,0])
            if (W[i+1,0] > 0):
                a_r = math.sqrt(g*W[i+1,0])
            (s_sr, s_hr, s_sl, s_hl) = f.get_dry_speeds(W[i,1], a_l, W[i+1,1], a_r)
            max_dry_speed = max(abs(s_sr), abs(s_hr), abs(s_sl), abs(s_hl))
            if(S_max < max_dry_speed):
                S_max = max_dry_speed 
    delta_t = CFL*dx/S_max
    return (delta_t, boundary_flux)

def flux_at_boundaries_data_driven_riemann(W, g, cells, dx, CFL, model): 
    S_max = -1.0
    # seperate into left and right states for riemann problem
    W_l = W[:-1,:]
    W_r = W[1:,:]
    # solve all the riemann problems predicting h_s
    h_s = riemann_aux.compute(model, W_l[:,0:-1], W_r[:,0:-1], 2)
    # control for dry states
    dry_right = W_r[:,0] <= 10e-7
    dry_left = W_l[:,0] <= 10e-7
    dry_midle = h_s <= 10e-7
    not_dry = np.logical_not(np.logical_or(np.logical_or(dry_right, dry_left), dry_midle))
    u_s = 1/2*(W_l[:,1] + W_r[:,1]) + 1/2*(f.f_k_numpy(g, h_s, W_r[:,0]) - f.f_k_numpy(g, h_s, W_l[:,0]))
    S_wet = np.max(np.absolute(W_l[not_dry,1]) + np.sqrt(g*W_l[not_dry,0]))
    S_dry = f.get_dry_speeds_np(W_l[~not_dry & ~dry_left,1], np.sqrt(g*W_l[~not_dry & ~dry_left,0]), W_r[~not_dry & ~dry_right,1], np.sqrt(g*W_r[~not_dry & ~dry_right,0]))
    if S_dry.size == 0:
        S_max = S_wet
    else:
        dry_max = np.max(np.absolute(S_dry))
        S_max = np.maximum(S_wet, dry_max)
    # types of waves
    right_shear = np.where(not_dry, u_s > 0, False) # right going shear wave  
    l_rarefaction1 = np.where(not_dry, np.logical_and(h_s <= W_l[:,0], (u_s-np.sqrt(g*h_s)) < 0), False)
    l_rarefaction2 = np.where(not_dry, np.logical_and(h_s <= W_l[:,0], (W_l[:,1]-np.sqrt(g*W_l[:,0])) > 0), False)
    l_sonic_rarefaction3 = np.where(not_dry, np.logical_and(np.logical_and(h_s <= W_l[:,0], ~l_rarefaction1), ~l_rarefaction2), False)
    l_shock_n = np.where(not_dry, np.logical_and(h_s > W_l[:,0], (W_l[:,1]-np.sqrt(g*W_l[:,0])*np.sqrt(1/2*((h_s+W_l[:,0])*h_s/(W_l[:,0]**2))) < 0)), False)
    l_shock_p = np.where(not_dry, np.logical_and(h_s > W_l[:,0], ~l_shock_n), False)
    r_rarefaction1 = np.where(not_dry, np.logical_and(h_s <= W_r[:,0], (u_s+np.sqrt(g*h_s)) > 0), False)
    r_rarefaction2 = np.where(not_dry, np.logical_and(h_s <= W_r[:,0], (W_r[:,1]+np.sqrt(g*W_r[:,0])) < 0), False)
    r_sonic_rarefaction3 = np.where(not_dry, np.logical_and(np.logical_and(h_s <= W_r[:,0], ~r_rarefaction1), ~r_rarefaction2), False)
    r_shock_p = np.where(not_dry, np.logical_and(h_s > W_r[:,0], W_r[:,1]+np.sqrt(g*W_r[:,0])*np.sqrt(1/2*((h_s+W_r[:,0])*h_s/(W_r[:,0]**2))) > 0), False)
    r_shock_n = np.where(not_dry, np.logical_and(h_s > W_r[:,0], ~r_shock_p), False)
    # content of waves at the boundaries
    Wb = np.empty([cells+1, 3])
    Wb[right_shear & (l_rarefaction1 | l_shock_n)] = np.array([h_s[right_shear & (l_rarefaction1 | l_shock_n)], u_s[right_shear & (l_rarefaction1 | l_shock_n)], W_l[right_shear & (l_rarefaction1 | l_shock_n),2]]).T
    Wb[right_shear & (l_rarefaction2 | l_shock_p)] = W_l[right_shear & (l_rarefaction2 | l_shock_p),:] 
    mask_lsonic = right_shear & l_sonic_rarefaction3 | (dry_right & ~dry_left)
    Wb[mask_lsonic] = np.array([(((W_l[mask_lsonic,1]+2*np.sqrt(g*W_l[mask_lsonic,0]))/3)**2)/g, (W_l[mask_lsonic,1]+2*np.sqrt(g*W_l[mask_lsonic,0]))/3, W_l[mask_lsonic,2]]).T
    Wb[~right_shear & (r_rarefaction1 | r_shock_p)] = np.array([h_s[~right_shear & (r_rarefaction1 | r_shock_p)], u_s[~right_shear & (r_rarefaction1 | r_shock_p)], W_r[~right_shear & (r_rarefaction1 | r_shock_p),2]]).T
    Wb[~right_shear & (r_rarefaction2 | r_shock_n)] = W_r[~right_shear & (r_rarefaction2 | r_shock_n),:] 
    mask_rsonic = ~right_shear & r_sonic_rarefaction3 | (~dry_right & dry_left)
    Wb[mask_rsonic] = np.array([(((2*np.sqrt(g*W_r[mask_rsonic,0]) -W_r[mask_rsonic,1])/3)**2)/g, (W_r[mask_rsonic,1]-2*np.sqrt(g*W_r[mask_rsonic,0]))/3, W_r[mask_rsonic,2]]).T 
    Wb[~not_dry & ~mask_lsonic & ~mask_rsonic] = np.array([0,0,0])
    # fluxes
    fluxes = np.array([Wb[:,0]*Wb[:,1], Wb[:,0]*(Wb[:,1]**2) + 0.5*g*Wb[:,0]**2, Wb[:,0]*Wb[:,1]*Wb[:,2]]).transpose()
    delta_t = CFL*dx/S_max
    return (delta_t, fluxes)

def flux_at_boundaries_data_driven_god(W, g, dx, CFL, model): 
    # seperate into left and right states for riemann problem
    W_l = W[:-1,:]
    W_r = W[1:,:]
    flux = np.empty([W.shape[0]-1, 3])
    flux[:,:-1] = lax_godunov_flux_aux.compute(model, W_l[:,0:-1], W_r[:,0:-1])
    mask_flux = W_l[:,0] <= 10e-8
    flux[mask_flux,:] = 0
    flux[:,-1] = 0.0 
    dry_right = W_r[:,0] <= 10e-8
    W_r[dry_right,:] = 0
    dry_left = W_l[:,0] <= 10e-8
    W_l[dry_left,:] = 0
    not_dry = np.logical_not(np.logical_or(dry_right, dry_left))
    S_wet = np.max(np.absolute(W_l[not_dry,1]) + np.sqrt(g*W_l[not_dry,0]))
    S_dry = f.get_dry_speeds_np(W_l[~not_dry & ~dry_left,1], np.sqrt(g*W_l[~not_dry & ~dry_left,0]), W_r[~not_dry & ~dry_right,1], np.sqrt(g*W_r[~not_dry & ~dry_right,0]))
    if S_dry.size == 0:
        S_max = S_wet
    else:
        dry_max = np.max(np.absolute(S_dry))
        S_max = np.maximum(S_wet, dry_max)
    delta_t = CFL*dx/S_max
    return (delta_t, flux)

""" This function returns (S_type, wave_speeds, h_s, u_s, boundary_w), used in the TVD-WAF scheme
u_s, h_s is the values in the star region. Boundary_w is the primite values at the interfaces.
The wave_speeds, is an array of 3 values, the speed of each wave, note that some type of wave patterns do not have 3 different waves. 
S_type, described what types of waves we are dealing with. Each of the 11 possible wave types are listed below:
0 = Same level on both sides of interface, means no waves.
1 = Both sides of interface are dry, all values are 0 
2 = Dry bed, dry-left, wet-right wave_speeds[0] is S_sr, wave_speeds[1] is S_r
3 = Dry bed, wet-left, dry-right wave_speeds[0] is S_l, wave_speeds[1] is S_sl 
4 = Dry bed, dry midel, wet-left, wet-right wave_speeds[0] is S_l, wave_speeds[1] is S_r
5 = Wet bed, with critical left rarefaction wave_speeds[0] is S_hl, wave_speeds[1] u_s, wave_speeds[2] is S_r
6 = Wet bed, with critical right rarefaction wave_speeds[0] is S_l, wave_speeds[1] u_s, wave_speeds[2] is S_hr
7 = Wet bed, shock-shock wave_speeds[0] is S_l, wave_speeds[1] is u_s, wave_speeds[2] is S_r
8 = Wet bed, rarefaction-shock wave_speeds[0] is S_hl, wave_speeds[1] is u_s, wave_speeds[2] is S_r 
9 = Wet bed, shock-rarefaction wave_speeds[0] is S_l, wave_speeds[1] is u_s, wave_speeds[2] is S_hr
10 = wet bed, rarefaction-rarefaction wave_speeds[0] is S_hl, wave_speeds[1] is u_s, wave_speeds[2] is S_hr
"""

def interface_wave_speeds(W_l, W_r, g, riemann_int, tolenrance, iterations):
    if np.all(W_l == W_r): # Case 0
        return (0, np.array([0.0, 0.0, 0.0]), W_l[0].item(), W_l[1].item(), W_l)
    dry, _, h_u_s = False, None, None # Just for scoping reasons
    if (riemann_int == 0):
        (dry, W_x, _, h_u_s) = exact_riemann_solver.solve(0.0, W_l, W_r, g, tolenrance, iterations)
    if (riemann_int == 1):
        (dry, W_x, _, h_u_s) = hllc_riemann.solve(W_l, W_r, g)
    if (dry):
        if (W_l[0] <= 0 and W_r[0] <= 0): # Case 1
            return (1, np.array([0.0, 0.0, 0.0]), h_u_s[0], h_u_s[1], W_x)
        elif (W_l[0] <= 0): # Case 2
            return (2, np.array([W_r[1]-2*math.sqrt(W_r[0]*g), W_r[1]+math.sqrt(W_r[0]*g), 0.0]), h_u_s[0], h_u_s[1], W_x)
        elif (W_r[0] <= 0): # Case 3
            return (3, np.array([W_l[1]-math.sqrt(W_l[0]*g), W_l[1]+2*math.sqrt(W_l[0]*g), 0.0]), h_u_s[0], h_u_s[1], W_x)
        else: # Case 4
            return (4, np.array([W_l[1]-math.sqrt(W_l[0]*g), W_r[1]+math.sqrt(W_r[0]*g), 0.0]), 0.0, 0.0, np.array([0.0, 0.0, 0.0]))
    else: 
        if (h_u_s[1] > 0 and h_u_s[0] <= W_l[0] and not(np.sign(W_l[1] - math.sqrt(g*W_l[0])) == np.sign(h_u_s[1]-math.sqrt(h_u_s[0]*g)))): # Case 5
            return (5, np.array([W_l[1]-math.sqrt(W_l[0]*g), h_u_s[1],  W_r[1]+math.sqrt(W_r[0]*g)*f.qk(h_u_s[0], W_r[0])]), h_u_s[0], h_u_s[1], W_x)
        elif(h_u_s[1] < 0 and h_u_s[0] <= W_r[0] and not(np.sign(W_r[1] + math.sqrt(W_r[0]*g)) == np.sign(h_u_s[1]+math.sqrt(h_u_s[0]*g)))): # Case 6
            return (6, np.array([W_l[1]-math.sqrt(W_l[0]*g)*f.qk(h_u_s[0], W_l[0]), h_u_s[1],  W_r[1]+math.sqrt(W_r[0]*g)]), h_u_s[0], h_u_s[1], W_x)
        elif(h_u_s[0] > W_l[0] and h_u_s[0] > W_r[0]): # Case 7
            return (7, np.array([W_l[1]-math.sqrt(W_l[0]*g)*f.qk(h_u_s[0], W_l[0]), h_u_s[1],  W_r[1]+math.sqrt(W_r[0]*g)*f.qk(h_u_s[0], W_r[0])]), h_u_s[0], h_u_s[1], W_x)
        elif(h_u_s[0] <= W_l[0] and h_u_s[0] > W_r[0]): # Case 8
            return (8, np.array([W_l[1]-math.sqrt(g*W_l[0]), h_u_s[1],  W_r[1]+math.sqrt(W_r[0]*g)*f.qk(h_u_s[0], W_r[0])]), h_u_s[0], h_u_s[1], W_x)
        elif(h_u_s[0] > W_l[0] and h_u_s[0] <= W_r[0]): # Case 9
            return (9, np.array([W_l[1]-math.sqrt(W_l[0]*g)*f.qk(h_u_s[0], W_l[0]), h_u_s[1],  W_r[1]+math.sqrt(g*W_r[0])]), h_u_s[0], h_u_s[1], W_x)
        else: # Case 10
            return (10, np.array([W_l[1]-math.sqrt(g*W_l[0]), h_u_s[1],  W_r[1]+math.sqrt(g*W_r[0])]), h_u_s[0], h_u_s[1], W_x)

# This method returns (number of shock/rarefaction waves, list_of_c_for_each_wave, list_of_jump_in_w, list_of_jump_in_flux) 
def get_c_dw_dflux(S_type, S_array, W_l, W_r, delta_t, delta_x, boundary_flux, boundary_w, h_s, u_s, g):
    no_output = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    zero_array = np.array([0.0, 0.0, 0.0])
    match S_type:
        case 0: # W is the same on both sides of boundary
            return (0, zero_array, no_output, no_output) 
        case 1: # Both sides of boundary are dry
            return (0, zero_array, no_output, no_output) 
        case 2: # Left side is dry
            delta_w_1 = boundary_w - zero_array # w jump across wave 1 
            delta_w_2 = W_r - boundary_w  # w jump across shear wave
            delta_flux_1 = boundary_flux - zero_array  # flux jump across wave 1 
            delta_flux_2 = f.flux_from_w(W_r, g) - boundary_flux  # flux jump across wave 1 
            return (2, np.array([delta_t/delta_x*S_array[0], delta_t/delta_x*S_array[1], 0.0]), np.array([delta_w_1, delta_w_2, zero_array]), np.array([delta_flux_1, delta_flux_2, zero_array]))
        case 3: # Right side is dry
            delta_w_1 = boundary_w - W_l # w jump across wave 1 
            delta_w_2 = zero_array - boundary_w  # w jump across shear wave
            delta_flux_1 = boundary_flux - f.flux_from_w(W_l, g) # flux jump across wave 1 
            delta_flux_2 = zero_array - boundary_flux # flux jump across shear wave
            return (2, np.array([delta_t/delta_x*S_array[0], delta_t/delta_x*S_array[1], 0.0]), [delta_w_1, delta_w_2, zero_array], [delta_flux_1, delta_flux_2, zero_array])
        case 4: # Both sides of boundary are wet, but dry region is created in between
            delta_w_1 = zero_array - W_l # w jump across wave 1 
            delta_w_2 = W_r - zero_array # w jump across wave 2 
            delta_flux_1 = zero_array - f.flux_from_w(W_l, g) # flux jump across wave 1 
            delta_flux_2 = f.flux_from_w(W_r, g) - zero_array # flux jump across wave 2 
            return (2, np.array([delta_t/delta_x*S_array[0], delta_t/delta_x*S_array[1], 0.0]), np.array([delta_w_1, delta_w_2, zero_array]), np.array([delta_flux_1, delta_flux_2, zero_array]))
        case 5: # Left sonic rarefaction
            delta_w_1 = boundary_w - W_l # w jump across wave 1 
            delta_w_2 = np.array([h_s, u_s, W_r[2]]) - boundary_w # w jump across wave 2
            delta_w_3 = W_r - np.array([h_s, u_s, W_r[2]]) # w jump across wave 3
            delta_flux_1 = boundary_flux - f.flux_from_w(W_l, g) # flux jump across wave 1 
            delta_flux_2 = f.flux_from_w(np.array([h_s, u_s, W_r[2]]), g) - boundary_flux # flux jump across wave 2 
            delta_flux_3 = f.flux_from_w(W_r, g) - f.flux_from_w(np.array([h_s, u_s, W_r[2]]), g) # flux jump across wave 3 
            return (3, np.array([delta_t/delta_x*S_array[0], delta_t/delta_x*S_array[1], delta_t/delta_x*S_array[2]]), np.array([delta_w_1, delta_w_2, delta_w_3]), np.array([delta_flux_1, delta_flux_2, delta_flux_3]))
        case 6: # right sonic rarefaction
            delta_w_1 = np.array([h_s, u_s, W_l[2]]) - W_l # w jump across wave 1 
            delta_w_2 = boundary_w - np.array([h_s, u_s, W_l[2]]) # w jump across wave 2
            delta_w_3 = W_r - boundary_w # w jump across wave 3
            delta_flux_1 = f.flux_from_w(np.array([h_s, u_s, W_l[2]]), g) - f.flux_from_w(W_l, g) # flux jump across wave 1 
            delta_flux_2 = boundary_flux - f.flux_from_w(np.array([h_s, u_s, W_l[2]]), g) # flux jump across wave 2 
            delta_flux_3 = f.flux_from_w(W_r, g) - boundary_flux # flux jump across wave 3 
            return (3, np.array([delta_t/delta_x*S_array[0], delta_t/delta_x*S_array[1], delta_t/delta_x*S_array[2]]), np.array([delta_w_1, delta_w_2, delta_w_3]), np.array([delta_flux_1, delta_flux_2, delta_flux_3]))
        case other: # Any other case with 3 waves (shock-shock, shock-rarefaction, rarefaction-shock, rarefaction-rarefaction)
            delta_w_1 = np.array([h_s, u_s, W_l[2]]) - W_l # w jump across wave 1 
            delta_w_2 = np.array([h_s, u_s, W_r[2]]) - np.array([h_s, u_s, W_l[2]]) # w jump across wave 2
            delta_w_3 = W_r - np.array([h_s, u_s, W_r[2]]) # w jump across wave 3
            delta_flux_1 = f.flux_from_w(np.array([h_s, u_s, W_l[2]]), g) - f.flux_from_w(W_l, g) # flux jump across wave 1 
            delta_flux_2 = f.flux_from_w(np.array([h_s, u_s, W_r[2]]), g) - f.flux_from_w(np.array([h_s, u_s, W_l[2]]), g) # flux jump across wave 2 
            delta_flux_3 = f.flux_from_w(W_r, g) - f.flux_from_w(np.array([h_s, u_s, W_r[2]]), g) # flux jump across wave 3 
            return (3, np.array([delta_t/delta_x*S_array[0], delta_t/delta_x*S_array[1], delta_t/delta_x*S_array[2]]), np.array([delta_w_1, delta_w_2, delta_w_3]), np.array([delta_flux_1, delta_flux_2, delta_flux_3]))
    print("Error: this should never happen an error occured in the get_c_dw_dflux function")

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
        
def flux_waf_tvd(W, g, riemann_int, cells, delta_t, delta_x, boundary_flux, tolenrance, iterations):
    S_type, S_array, h_u_s, boundary_w_array = np.empty(cells+1, dtype=int), np.empty((cells+1,3)), np.empty((cells+1,2)), np.empty((cells+1,3))
    n_waves_array, c, w_jumps, delta_flux, waf_flux = np.empty(cells+1, dtype=int), np.empty((cells+1,3)), np.empty((cells+1,3,3)), np.empty((cells+1,3,3)), np.empty((cells+1,3))
    for i in range(cells+1):
        S_type[i], S_array[i], h_u_s[i,0], h_u_s[i,1], boundary_w_array[i] = interface_wave_speeds(W[i,:], W[i+1,:], g, riemann_int, tolenrance, iterations)
    for i in range(cells+1):
        n_waves_array[i], c[i], w_jumps[i], delta_flux[i] = get_c_dw_dflux(S_type[i], S_array[i], W[i,:], W[i+1,:], delta_t, delta_x, boundary_flux[i], boundary_w_array[i], h_u_s[i,0], h_u_s[i,1], g)
    for i in range(cells+1):
        waf_flux_without_sum = 0.5*(f.flux_from_w(W[i], g) + f.flux_from_w(W[i+1], g))
        waf_sum = np.array([0.0, 0.0, 0.0])
        for j in range(n_waves_array[i]):
            if j == 0 or j == 2 or (j == 1 and n_waves_array[i] == 2):
                if (math.isclose(w_jumps[i,j,0], 0.0, rel_tol=1e-8)):
                    continue
                waf_sum = waf_sum + np.sign(c[i,j].item())*super_bee_limiter(w_jumps[i-1,j,0], w_jumps[i,j,0], w_jumps[i+1,j,0], c[i,j])*delta_flux[i,j]
            else:
                if (math.isclose(w_jumps[i,j,2], 0.0, rel_tol=1e-8)):
                    continue
                waf_sum = waf_sum + np.sign(c[i,j])*super_bee_limiter(w_jumps[i-1,j,2], w_jumps[i,j,2], w_jumps[i+1,j,2], c[i,j])*delta_flux[i,j]
        waf_flux[i] = (waf_flux_without_sum - (1/2*waf_sum))
    return waf_flux 
