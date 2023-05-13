import matplotlib as mpl
import matplotlib.pyplot as plt
from IPython.display import display, clear_output # this is for plotting animations
import numpy as np

# this function is used if one wants to plot the numerical scheme as it progresses
def animator(figuare, ax):
    display(figuare)
    clear_output(wait=True)
    plt.pause(2.0)

def make_plot(out_plot_name, out_path, save, x, t_end, tuple_bool_exact_scatter, exact_data, scheme, numerical_data, figuare, ax, riemann_str, h_u_psi_str):
    mpl.rcParams['font.size'] = mpl.rcParams['font.size']*0.8 
    if scheme >= 0:
        ax.hlines(numerical_data[1:-2], x[0:-2], x[1:-1], colors=['black'], linewidth=1.0, label='numerical solution')
    if tuple_bool_exact_scatter[0]:
        ax.plot(x, exact_data, linewidth=0.5, label='exact solution')
        if (tuple_bool_exact_scatter[1]):
            ax.scatter(x, exact_data, marker='o', facecolors='white', color='k', s=1, label='exact solution points')
            figuare.suptitle(h_u_psi_str + ", exact solution at t = " + str(t_end) + " to " + out_plot_name)
    if scheme == 0: # godunov upwind method
        figuare.suptitle(h_u_psi_str + ", at t = " + str(t_end) + ", " + out_plot_name + " using Godunov upwind, and " + riemann_str + " Riemann solver")
    if scheme == 1: # lax-friedrichs
        figuare.suptitle(h_u_psi_str + ", at t = " + str(t_end) + ", " + out_plot_name + " using Lax-Friedrich")
    if scheme == 2: # WAF scheme 
        figuare.suptitle(h_u_psi_str + ", at t = " + str(t_end) + ", " + out_plot_name + " using WAF, and " + riemann_str + " Riemann solver")
    ax.set_ylabel("height of water " + h_u_psi_str, fontsize=8)
    ax.set_xlabel("length of the channel, x", fontsize=8)
    ax.xaxis.get_label().set_x(0.5)
    ax.xaxis.get_label().set_y(-0.1)
    ax.yaxis.get_label().set_x(-0.1)
    ax.yaxis.get_label().set_y(0.5)
    plt.legend()
    figuare.subplots_adjust(wspace=0.3, bottom=0.2)
    if save:
        plt.savefig(out_path + '/' + h_u_psi_str + out_plot_name + riemann_str + ".png", dpi=300)

def plot(out_plot_name, out_path, animate, save, x_len, t_end, cells, tuple_bool_exact_scatter, exact_data, scheme, numerical_data, riemann_str):
    figure, ax = plt.subplots(1,1)

    x = np.linspace(0, x_len, cells+1)
    for i in range(3):
        temp_str = ""
        if i == 0:
            temp_str = "h(x)"
        elif i == 1:
            temp_str = "u(x)"
        elif i == 2:
            temp_str = "psi(x)"
        make_plot(out_plot_name, out_path, save, x, t_end, tuple_bool_exact_scatter, exact_data[:,i], scheme, numerical_data[:,i], figure, ax, riemann_str, temp_str)
        if animate:
            animator(figure, ax)
        ax.clear()
        
def plot_error_and_speed(speed_data, error_data, h, cells, order, out_plot_name, out_path, scheme, riemann_str):
    mpl.rcParams['font.size'] = mpl.rcParams['font.size']*0.5 
    figuare, ax = plt.subplots(1,2)
    ax[0].loglog(h, error_data, '-o', label='global error', markersize=1.6, linewidth=0.9)
    # make a slop of 1 in log-log plot 
    slope_y = [4*((error_data[0])*((1/2)**i)) for i in range(len(h))]
    label_str = "slope -" + str(order) + " order"
    ax[0].loglog(h, slope_y, '-o', color='red', label=label_str, markersize=1.6, linewidth=0.9)
    ax[0].set_xlabel("h", fontsize=8)
    ax[0].set_ylabel("error in meters", fontsize=8)
    if (scheme == "Lax Friedrich"):
        ax[0].set_title("Scheme: " + scheme + ", " + out_plot_name)
    else:     
        ax[0].set_title("Scheme: " + scheme + ", " + out_plot_name + " using " + riemann_str + " Riemann solver")
    ax[0].legend()
    slope_y = [4*((speed_data[0])*((4)**i)) for i in range(len(cells))] #for each doubling on the number of cells, we expect 4 times slower computation giving O(cells^2)
    ax[1].loglog(cells, speed_data, '-o', label='compute time', markersize=1.6, linewidth=0.9)
    ax[1].plot(cells, slope_y, '-o', color='red', label="slope $O(n^2)$", markersize=1.6, linewidth=0.9)
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("cells", fontsize=8)
    ax[1].set_ylabel("time in seconds", fontsize=8)
    if (scheme == "Lax Friedrich"):
        ax[1].set_title("Scheme: " + scheme + ", " + out_plot_name)
    else:     
        ax[1].set_title("Scheme: " + scheme + ", " + out_plot_name + " using " + riemann_str + " Riemann solver")
    ax[1].legend()
    figuare.subplots_adjust(wspace=0.3, bottom=0.2)
    plt.savefig(out_path + '/' + scheme + "_" + out_plot_name + '_' + riemann_str + ".png", dpi=300)

