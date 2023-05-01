import math
import sys
from aux_functions import exact_riemann_solver

def flux_riemann(riemann_solutions, g, cells):
    fluxes_at_boundaries = [[],[],[]]
    # define u1-3 for scope
    for i in range(cells+1):
        u1 = riemann_solutions[0][i]
        if (u1 <= 0):
            fluxes_at_boundaries[0].append(0.0)
            fluxes_at_boundaries[1].append(0.0)
            fluxes_at_boundaries[2].append(0.0)
            continue
        u2 = u1 * riemann_solutions[1][i]
        u3 = u1 * riemann_solutions[2][i]
        fluxes_at_boundaries[0].append(u2)
        fluxes_at_boundaries[1].append((u2**2)/u1 + 0.5*g*(u1**2))
        fluxes_at_boundaries[2].append((u2*u3)/u1)
    return fluxes_at_boundaries

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

def W_from_U(U, W, cells):
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

def riemann_interface(bool_output, out_file, W, g, cells, solver, x_len, tolerance, iteration, CFL): 
    S_max = -1.0
    riemann_solutions = [[],[],[]]
    for i in range(cells+1):
        if solver == 0: #exact
            (dry_bool, h_x, u_x, psi_x, max_dry_speed) = exact_riemann_solver.solve(bool_output, out_file, 0.0, W[0][i], W[1][i], W[2][i], W[0][i+1], W[1][i+1], W[2][i+1], g, tolerance, iteration)
            if not(dry_bool): #in the weet bed case
                riemann_solutions[0].append(h_x)
                riemann_solutions[1].append(u_x)
                riemann_solutions[2].append(psi_x)
                S = abs(W[1][i]) + math.sqrt(g*W[0][i])
                if(S_max < S):
                    S_max = S 
            else: #in the dry bed case
                riemann_solutions[0].append(h_x)
                riemann_solutions[1].append(u_x)
                riemann_solutions[2].append(psi_x)
                if(S_max < max_dry_speed):
                    S_max = max_dry_speed 
        elif solver == 1: # HLL
            print("not implemented yet, HLL Riemann")
            sys.exit(1)
        elif solver == 2: # HLLC 
            print("not implemented yet, HLLC Riemann")
            sys.exit(1)
    delta_x = x_len/cells
    delta_t = CFL*delta_x/S_max
    return (delta_t, riemann_solutions)

def center_dt_wet(W, cells, g, cfl, dx):
    S_max = -1.0
    for i in range(1,cells+1):
        S = abs(W[1][i]) + math.sqrt(g*W[0][i])
        if(S_max < S):
            S_max = S
    dt = cfl*dx/S_max
    return dt