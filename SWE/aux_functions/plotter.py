import matplotlib.pyplot as plt
from IPython.display import display, clear_output # this is for plotting animations
import numpy as np

# this function is used if one wants to plot the numerical scheme as it progresses
def animator(figuare, ax):
    display(figuare)
    clear_output(wait=True)
    plt.pause(2.0)

def make_plot(out_plot_name, out_path, save, x, t_end, tuple_bool_exact_scatter, exact_data, scheme, numerical_data, figuare, ax, riemann_str, h_u_psi_str):
    if scheme > 0:
        ax.hlines(numerical_data[1:-2], x[0:-2], x[1:-1], colors=['black'], linewidth=1.0)
    if tuple_bool_exact_scatter[0]:
        ax.plot(x, exact_data, linewidth=0.5)
        if (tuple_bool_exact_scatter[1]):
            ax.scatter(x, exact_data, marker='o', facecolors='white', color='k', s=1)
            figuare.suptitle(h_u_psi_str + ", exact solution at t = " + str(t_end) + " to " + out_plot_name)
    if scheme == 1: # godunov upwind upwind method
        figuare.suptitle(h_u_psi_str + ", at t = " + str(t_end) + ", " + out_plot_name + " using Godunov upwind, and " + riemann_str + " Riemann solver")
    if scheme == 2: # lax-friedrichs upwind method
        figuare.suptitle(h_u_psi_str + ", at t = " + str(t_end) + ", " + out_plot_name + " using Lax-Friedrich")
    if scheme == 3: # WAF scheme 
        figuare.suptitle(h_u_psi_str + ", at t = " + str(t_end) + ", " + out_plot_name + " using WAF, and " + riemann_str + " Riemann solver")
    ax.set_ylabel("height of water " + h_u_psi_str)
    ax.set_xlabel("length of the channel, x")
    if save:
        plt.savefig(out_path + '/' + h_u_psi_str + out_plot_name + ".png", dpi=300)

def plot(out_plot_name, out_path, animate, save, x_len, t_end, cells, tuple_bool_exact_scatter, exact_data, scheme, numerical_data, riemann_str):
    figuare, ax = plt.subplots(1,1)
    x = np.linspace(0, x_len, cells+1)
    for i in range(3):
        if i == 0:
            make_plot(out_plot_name, out_path, save, x, t_end, tuple_bool_exact_scatter, exact_data[i], scheme, numerical_data[i], figuare, ax, riemann_str, "h(x)")
        if i == 1:
            make_plot(out_plot_name, out_path, save, x, t_end, tuple_bool_exact_scatter, exact_data[1], scheme, numerical_data[1], figuare, ax, riemann_str, "u(x)")
        if i == 2:
            make_plot(out_plot_name, out_path, save, x, t_end, tuple_bool_exact_scatter, exact_data[2], scheme, numerical_data[2], figuare, ax, riemann_str, "psi(x)")
        if animate:
            animator(figuare, ax)
        ax.clear()
        