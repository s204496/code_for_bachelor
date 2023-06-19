"This file generates data, that help understand, who each of the parameters in the exact Riemann solver changes the function, that we are trying to appriximate. We are trying to find h_s"

import sys
sys.path.append('../SWE')
import numpy as np
import matplotlib.pyplot as plt
import math 
from aux_functions import wet_bed
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from data_driven.aux_function import general_aux, riemann_aux

def riemann_solver(h_l, u_l, h_r, u_r, g):
    W_out = wet_bed.calculate(g, 1e-8, 50, h_l, u_l, math.sqrt(g*h_l), h_r, u_r, math.sqrt(g*h_r))
    return W_out[0]

def riemann_solver_delta(h_l, h_r, g, delta_u):
    if delta_u < 0:
        u_l = 6
        u_r = 6 + delta_u
    else:
        u_l = 6 - delta_u
        u_r = 6
    W_out = wet_bed.calculate(g, 1e-8, 50, h_l, u_l, math.sqrt(g*h_l), h_r, u_r, math.sqrt(g*h_r))
    return W_out[0]

def plot_fixed_u(u_l, u_r, g):
    h_r = np.linspace(0.01, 3, 100)
    h_l = np.linspace(0.01, 3, 100)
    Z = np.zeros((len(h_l), len(h_r)))
    for i in range(len(h_l)):
        for j in range(len(h_r)):
            Z[i][j] = riemann_solver(h_l[i].item(), u_l, h_r[j].item(), u_r, g) 
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    H_L, H_R = np.meshgrid(h_l, h_r)
    ax.plot_surface(H_L, H_R, Z, cmap='viridis')
    ax.set_xlabel(r'$h_l$')
    ax.set_ylabel(r'$h_r$')
    ax.set_zlabel(r'$h_s$')
    plt.show()

def plot_half_h_fixed_u(u_l, u_r, g):
    h_r = np.linspace(0.01, 3, 100)
    Z = np.full((len(h_r), len(h_r)), np.nan)
    for i in range(len(h_r)):
        h_l = np.linspace(0.01, h_r[i], i+1)
        for j in range(len(h_l)):
            Z[i][j] = riemann_solver(h_l[j].item(), u_l, h_r[i].item(), u_r, g)
    plt.figure()
    ax = plt.axes(projection='3d')
    H_L, H_R = np.meshgrid(h_l, h_r)
    ax.contour3D(H_L, H_R, Z, levels=50, colors='black')
    #ax.plot_surface(H_L, H_R, Z, cmap='viridis')
    ax.set_xlabel('h_l')
    ax.set_ylabel('h_r')
    ax.set_zlabel('h_s')
    plt.show()
    
def plot_half_u_fixed_h(h_l, h_r, g):
    u_r = np.linspace(-6, 6, 100)
    Z = np.full((len(u_r), len(u_r)), np.nan)
    for i in range(len(u_r)):
        u_l = np.linspace(-6, u_r[i], i+1)
        for j in range(len(u_l)):
            Z[i][j] = riemann_solver(h_l, u_l[i].item(), h_r, u_r[j].item(), g) 
    plt.figure()
    ax = plt.axes(projection='3d')
    U_L, U_R = np.meshgrid(u_l, u_r)
    ax.contour3D(U_L, U_R, Z, levels=50, colors='black')
    ax.set_xlabel('u_l')
    ax.set_ylabel('u_r')
    ax.set_zlabel('h_s')
    plt.show()   

def plot_fixed_h(h_l, h_r, g):
    u_r = np.linspace(-6, 6, 100)
    u_l = np.linspace(-6, 6, 100)
    Z = np.zeros((len(u_l), len(u_r)))
    for i in range(len(u_l)):
        for j in range(len(u_r)):
            Z[i][j] = riemann_solver(h_l, u_l[i].item(), h_r, u_r[j].item(), g) 
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    U_L, U_R = np.meshgrid(u_l, u_r)
    ax.plot_surface(U_L, U_R, Z, cmap='viridis')
    ax.set_xlabel(r'$u_l$')
    ax.set_ylabel(r'$u_r$')
    ax.set_zlabel(r'$h_s$')
    plt.show()   

def plot_delta_h_u(g):
    delta_u = np.linspace(-12, 12, 15)
    h_r = np.linspace(0.01, 3, 15)
    Z = np.full((len(delta_u), len(h_r), len(h_r)), np.nan)
    for i in range(len(delta_u)):
        for j in range(len(h_r)):
            h_l = np.linspace(0.01, h_r[j], j+1)
            for z in range(len(h_l)):
                Z[i][j][z] = riemann_solver_delta(h_l[z].item(), h_r[j].item(), g, delta_u[i].item())
    _ = plt.figure()
    ax = plt.axes(projection='3d')
    Delta_U, H_R, H_L = np.meshgrid(delta_u, h_r, h_l)
    ax.set_xlabel(r'$\Delta$ u')
    ax.set_ylabel(r'$h_r$')
    ax.set_zlabel(r'$h_l$')
    ax.set_title(r'Color map of $h_s$')

    # Flatten the arrays for plotting
    x = Delta_U.flatten()
    y = H_R.flatten()
    z = H_L.flatten()
    c = Z.flatten()

    # Create the heatmap using scatter plot
    sc = ax.scatter(x, y, z, c=c, cmap='viridis', s=10)

    # Add colorbar
    plt.colorbar(sc)

    plt.show()

def plot_hat_hs_vs_hs(g):
    model = general_aux.load_model('data_driven/models/riemann_FFNN_shallow.pt', 'cpu', 'ffnn_riemann') # CPU can be changed if one has a Nvidia GPU
    data_input_l = np.zeros((1000, 2))
    data_input_r = np.zeros((1000, 2))
    h_s = np.zeros(1000)
    for i in range(1000):
        data_input_l[i][0] = np.random.uniform(0.01, 3)
        data_input_l[i][1] = np.random.uniform(-6, 6)
        data_input_r[i][0] = np.random.uniform(0.01, 3)
        data_input_r[i][1] = np.random.uniform(-6, 6)
        h_s[i] = riemann_solver(data_input_l[i][0], data_input_l[i][1], data_input_r[i][0], data_input_r[i][1], g)
    hat_hs = riemann_aux.compute(model, data_input_l[:,:], data_input_r[:,:], 2)
    _, ax = plt.subplots()
    ax.scatter(h_s, hat_hs, s=1)
    ax.set_xlabel(r'$h_s$')
    ax.set_ylabel(r'$\hat{h_s}$')
    ax.set_title(r'$\hat{h_s}$ vs $h_s$')
    # draw line from (0,0) to (7,7)
    plt.show()


def main():
    g = 9.8
    # test1
    #u_l = 6.0
    #u_r = -6.0
    #plot_fixed_u(u_l, u_r, g)
    # test2
    #h_l = 1.0 
    #h_r = 1.0
    #plot_fixed_h(h_l, h_r, g)
    # test 3, delta plot
    #plot_delta_h_u(g)
    # plot of \hat{h_s} vs h_s from exact solver
    plot_hat_hs_vs_hs(g)


if __name__ == '__main__':
    main()
    print("Data Generation - Completed successfully")
