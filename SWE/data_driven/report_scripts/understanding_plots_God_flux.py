"This file generates data, that help understand, who each of the parameters in Godunov_flux changes the function, that we are trying to appriximate. We are trying to find the flux of h and hu"

import sys
sys.path.append('../SWE')
import numpy as np
import matplotlib.pyplot as plt
import torch
from aux_functions import exact_riemann_solver
from data_driven.aux_function import general_aux, godunov_flux_aux

def god_flux(h_l, u_l, h_r, u_r, g):
    (_, (_, _, _), boundary_flux, _) = exact_riemann_solver.solve(0.0, np.array([h_l,u_l, 0.0]), np.array([h_r,u_r, 0.0]), g, 10E-8, 50)
    return boundary_flux[0], boundary_flux[1]

def plot_fixed_u(u_l, u_r, g):
    h_r = np.linspace(0.01, 3, 100)
    h_l = np.linspace(0.01, 3, 100)
    flux_h = np.zeros((len(h_l), len(h_r)))
    flux_hu = np.zeros((len(h_l), len(h_r)))
    for i in range(len(h_l)):
        for j in range(len(h_r)):
            flux_h[i][j], flux_hu[i][j] = god_flux(h_l[i].item(), u_l, h_r[j].item(), u_r, g) 
    _ = plt.figure()
    ax = plt.axes(projection='3d')
    H_L, H_R = np.meshgrid(h_l, h_r)
    ax.plot_surface(H_L, H_R, flux_h, cmap='viridis')
    ax.set_xlabel(r'$h_l$')
    ax.set_ylabel(r'$h_r$')
    ax.set_zlabel(r'$flux_{h}$')
    plt.show()
    _ = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(H_L, H_R, flux_hu, cmap='viridis')
    ax.set_xlabel(r'$h_l$')
    ax.set_ylabel(r'$h_r$')
    ax.set_zlabel(r'$flux_{hu}$')
    plt.show()

def plot_fixed_h(h_l, h_r, g):
    u_r = np.linspace(-6, 6, 100)
    u_l = np.linspace(-6, 6, 100)
    flux_h = np.zeros((len(u_l), len(u_r)))
    flux_hu = np.zeros((len(u_l), len(u_r)))
    for i in range(len(u_l)):
        for j in range(len(u_r)):
            flux_h[i][j], flux_hu[i][j] = god_flux(h_l, u_l[i].item(), h_r, u_r[j].item(), g) 
    _ = plt.figure()
    ax = plt.axes(projection='3d')
    U_L, U_R = np.meshgrid(u_l, u_r)
    ax.plot_surface(U_L, U_R, flux_h, cmap='viridis')
    ax.set_xlabel(r'$u_l$')
    ax.set_ylabel(r'$u_r$')
    ax.set_zlabel(r'$flux_{h}$')
    plt.show()
    _ = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(U_L, U_R, flux_hu, cmap='viridis')
    ax.set_xlabel(r'$u_l$')
    ax.set_ylabel(r'$u_r$')
    ax.set_zlabel(r'$flux_{hu}$')
    plt.show()
  
def plot_hat_flux_vs_flux(g):
    model = general_aux.load_model('data_driven/models/godunov_flux_exact_200k.pt', 'cpu', 'godunov_flux') # CPU can be changed if one has a Nvidia GPU
    data_input_l = np.zeros((1000, 2))
    data_input_r = np.zeros((1000, 2))
    fluxes = np.zeros((1000,2))
    for i in range(1000):
        data_input_l[i][0] = np.random.uniform(0.01, 3)
        data_input_l[i][1] = np.random.uniform(-6, 6)
        data_input_r[i][0] = np.random.uniform(0.01, 3)
        data_input_r[i][1] = np.random.uniform(-6, 6)
        fluxes[i][0], fluxes[i][1] = god_flux(data_input_l[i][0], data_input_l[i][1], data_input_r[i][0], data_input_r[i][1], g)
    hat_fluex = godunov_flux_aux.compute(model, data_input_l, data_input_r)

    _, (ax1, ax2) = plt.subplots(1, 2,figsize=(10, 5))

    ax1.scatter(fluxes[:, 0], hat_fluex[:, 0], s=1)
    ax2.scatter(fluxes[:, 1], hat_fluex[:, 1], s=1)
    ax1.set_xlabel(r'$flux_{h}$')
    ax1.set_ylabel(r'$\hat{flux}_h$')
    ax2.set_xlabel(r'$flux_{hu}$')
    ax2.set_ylabel(r'$\hat{flux}_{hu}$')

    plt.show()

def main():
    g = 9.8
    # test1
    #u_l = 1
    #u_r = -2
    #plot_fixed_u(u_l, u_r, g)
    # test2
    #h_l = 1.0 
    #h_r = 0.1
    #plot_fixed_h(h_l, h_r, g)
    # plot prediction vs true
    plot_hat_flux_vs_flux(g)

if __name__ == '__main__':
    main()
    print("Understanding plots successfully completed")
