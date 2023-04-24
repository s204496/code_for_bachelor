import matplotlib.pyplot as plt
import numpy as np

def plot(out_plot_name, sol_data, x_len, break_pos, t_end, cells):
    h_list = []
    u_list = []
    for (_,_,h,u) in sol_data:
        h_list.append(h)
        u_list.append(u)
    x = np.linspace(0, x_len, cells+1)
    figuare, ax = plt.subplots(1,2)
    ax[0].plot(x, h_list, linewidth=0.5)
    ax[0].scatter(x, h_list, marker='o', facecolors='white', color='k', s=1)
    ax[0].set_ylabel("height of water h(x)")
    ax[0].set_xlabel("length of the channel, x")
    ax[1].plot(x, u_list, linewidth=0.5)
    ax[1].scatter(x, u_list, marker='o', facecolors='white', color='k', s=1)
    ax[1].set_ylabel("particle velocity u(x)")
    ax[1].set_xlabel("length of the channel, x")
    figuare.tight_layout(pad=3.0)
    figuare.suptitle("Exact solution at t = " + str(t_end) + " to " + out_plot_name + " with initial dam break position = " + str(break_pos))
    plt.savefig('exact_riemann_solver/output/' + out_plot_name + ".png", dpi=300)

