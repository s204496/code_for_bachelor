import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# This function plots and saves the numerical solution at time t_end toghter with the exact solution.
def make_plot(out_plot_name, out_path, x_center, x_interval, t_end, tuple_bool_exact_scatter, exact_data, scheme, numerical_data, figuare, ax, riemann_str, y_label, plot_title, save_name, data_driven):
    mpl.rcParams['font.size'] = mpl.rcParams['font.size']*0.8 
    if scheme >= 0:
        if data_driven:
            ax.hlines(numerical_data[1:-1], x_interval[0:-1], x_interval[1:], colors=['black'], linewidth=1.0, label='data driven solution')
        else:
            ax.hlines(numerical_data[1:-1], x_interval[0:-1], x_interval[1:], colors=['black'], linewidth=1.0, label='numerical solution')
    if tuple_bool_exact_scatter[0]:
        ax.plot(x_center, exact_data, linewidth=0.5, label='exact solution')
        if (tuple_bool_exact_scatter[1]):
            ax.scatter(x_center, exact_data, marker='o', facecolors='white', color='k', s=1, label='exact solution points')
            figuare.suptitle(plot_title + ", exact solution at t = " + str(t_end) + " to " + out_plot_name, y = 0.93, fontsize=9.5)
    if scheme == 0: # godunov upwind method
        figuare.suptitle(plot_title + ", at t = " + str(t_end) + ", " + out_plot_name + " using Godunov upwind, and " + riemann_str + " Riemann solver", y = 0.93, fontsize=9.5)
    if scheme == 1: # lax-friedrichs
        if data_driven:
            figuare.suptitle(plot_title + ", at t = " + str(t_end) + ", " + out_plot_name + " using Lax-Friedrich, Data-driven", y = 0.93, fontsize=9.5)
        else:
            figuare.suptitle(plot_title + ", at t = " + str(t_end) + ", " + out_plot_name + " using Lax-Friedrich", y = 0.93, fontsize=9.5)
    if scheme == 2: # WAF scheme 
        figuare.suptitle(plot_title + ", at t = " + str(t_end) + ", " + out_plot_name + " using WAF, and " + riemann_str + " Riemann solver", y = 0.93, fontsize=9.5)
    ax.set_ylabel(y_label, fontsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_xlabel("x in m", fontsize=8)
    ax.tick_params(axis='x', labelsize=8)
    ax.xaxis.get_label().set_x(0.5)
    ax.xaxis.get_label().set_y(-0.1)
    ax.yaxis.get_label().set_x(-0.1)
    ax.yaxis.get_label().set_y(0.5)
    plt.legend()
    figuare.subplots_adjust(wspace=0.3, bottom=0.2)
    plt.savefig(out_path + '/' + save_name + out_plot_name + riemann_str + ".png", dpi=300)

# This method plots the numerical and exact solution for h, u, and psi in 3 different plots by calling make_plot
def plot(out_plot_name, out_path, x_len, t_end, cells, tuple_bool_exact_scatter, exact_data, scheme, numerical_data, riemann_str, data_driven):
    figure, ax = plt.subplots(1,1)
    dx = x_len/cells
    x_center = np.linspace(dx, dx*cells, cells)
    x_interval = np.linspace(0, x_len, cells+1)
    for i in range(3):
        plot_title = ""
        y_label = ""
        save_name = ""
        if i == 0:
            y_label = "h(x) in m"
            plot_title = "h(x)"
        elif i == 1:
            y_label = r'u(x) in $\frac{m}{s}$'
            plot_title = "u(x)"
        elif i == 2:
            y_label = r'$\psi(x)$ in $\frac{g}{m}$'
            plot_title = r'$\psi(x)$'
        if not(i == 2):
            save_name = plot_title
        else:
            save_name = "psi(x)"
        if np.any(numerical_data):
            make_plot(out_plot_name, out_path, x_center, x_interval, t_end, tuple_bool_exact_scatter, exact_data[:,i], scheme, numerical_data[:,i], figure, ax, riemann_str, y_label, plot_title, save_name, data_driven)
        else:
            make_plot(out_plot_name, out_path, x_center, x_interval, t_end, tuple_bool_exact_scatter, exact_data[:,i], scheme, numerical_data, figure, ax, riemann_str, y_label, plot_title, save_name, data_driven)
        ax.clear()
        
# Make a 2x2 plot for error of h, u, and psi, and speed of computation
def plot_error_and_speed(speed_data, error_data, delta_x_list, cells, out_plot_name, out_path, scheme, riemann_str):
    mpl.rcParams['font.size'] = mpl.rcParams['font.size']*0.5 
    figuare, ax = plt.subplots(2,2)
    for i in range(2):
        for j in range(2):
            if i == 1 and j == 1:
                break
            temp_str = ""
            if i == 0 and j ==0:
                temp_str = "h"
            elif i == 0 and j == 1:
                temp_str = "u"
            elif i == 1:
                temp_str = r'$\psi$'
            ax[i][j].loglog(delta_x_list, error_data[(i*2)+j], '-o', label=temp_str + ' error', markersize=1.6, linewidth=0.9)
            # make a slop of 1 in log-log plot 
            slope_y = [4*max(max(error_data[(i*2)+j]),0.001)*((1/2)**z) for z in range(len(delta_x_list))]
            label_str = "slope -" + str(1) + " order"
            ax[i][j].loglog(delta_x_list, slope_y, '-o', color='red', label=label_str, markersize=1.6, linewidth=0.9)
            ax[i][j].set_xlabel(r'$\Delta$x', fontsize=6)
            ax[i][j].set_ylabel(temp_str + " error", fontsize=6)
            if (scheme == "lax_friedrich"):
                ax[i][j].set_title("Scheme: Lax Friedrich, " + out_plot_name + ", error of " + temp_str)
            elif (scheme == "godunov_upwind"):     
                if riemann_str == 'riemann_data_driven':
                    ax[i][j].set_title("Scheme: Godunov upwind, " + out_plot_name + ", FFNN Riemann" + ", error of " + temp_str)
                elif riemann_str == 'data_driven_flux':
                    ax[i][j].set_title("Scheme: Godunov upwind, " + out_plot_name + ", FFNN flux" + ", error of " + temp_str)
                else:
                    ax[i][j].set_title("Scheme: Godunov upwind, " + out_plot_name + ", " + riemann_str + " solver" + ", error of " + temp_str)
            elif (scheme == "tvd_waf"):     
                ax[i][j].set_title("Scheme: TVD WAF, " + out_plot_name + ", " + riemann_str + " solver" + ", error of " + temp_str)
            ax[i][j].legend()
    slope_y = [4*((speed_data[0])*((4)**i)) for i in range(len(cells))] #for each doubling on the number of cells, we expect 4 times slower computation giving O(cells^2)
    ax[1][1].loglog(cells, speed_data, '-o', label='compute time', markersize=1.6, linewidth=0.9)
    ax[1][1].plot(cells, slope_y, '-o', color='red', label="slope $O(n^2)$", markersize=1.6, linewidth=0.9)
    ax[1][1].set_xscale("log")
    ax[1][1].set_yscale("log")
    ax[1][1].set_xlabel("cells", fontsize=6)
    ax[1][1].set_ylabel("time in seconds", fontsize=6)
    if (scheme == "lax_friedirhc"):
        ax[1][1].set_title("Scheme: Lax Friedrich, " + out_plot_name + ", speed")
    elif (scheme == "godunov_upwind"):
        ax[1][1].set_title("Scheme: Godunov Upwind, " + out_plot_name + ", " + riemann_str + " solver" + ", speed")
    elif (scheme == "tvd_waf"):
        ax[1][1].set_title("Scheme: TVD WAF, " + out_plot_name + ", " + riemann_str + " solver" + ", speed")
    ax[1][1].legend()
    figuare.subplots_adjust(wspace=0.4, bottom=0.3)
    plt.subplots_adjust(hspace=0.6)

    if (scheme == "lax_friedrich"):
        plt.savefig(out_path + '/' + scheme + "_" + out_plot_name + ".png", dpi=300)
    else:     
        plt.savefig(out_path + '/' + scheme + "_" + out_plot_name + '_' + riemann_str + ".png", dpi=300)


# Plotting the results of experiment 1 Riemann solver data driven models
def plot_experiment_1(models):
    colors = ['blue', 'green', 'red']
    titles = ['avg. MSE - loss', 'avg. Custom + MSE - loss']

    fig, axes = plt.subplots(2, 1, figsize=(8, 12))

    model_labels = ['Shallow', 'Deep', 'Custom loss']

    ap_indices = [0, 1, 2]

    for i in range(2):
        for model_idx, model_label in zip(ap_indices, model_labels):
            axes[i].plot(models[model_idx][i], color=colors[model_idx], linestyle='-', linewidth=1, label=model_label)
        
        axes[i].set_xlabel('Epochs')
        axes[i].set_ylabel('Loss')
        axes[i].set_yscale('log')
        axes[i].legend()
        axes[i].set_title(titles[i])

    plt.tight_layout()
    plt.savefig("output/data_driven/training/riemann_over_training.png")
    plt.show()

def plot_experiment_2(models, best_learning_rate, samples, out_str):
    color = ['blue', 'red']
    types = ['test', 'train']
    
    fig, axes = plt.subplots(1,1)

    for i in range(2):
        axes.plot(models[i], color=color[i], linestyle='-', linewidth=1, label=('MSE - ' + types[i] + ' loss'))
        
    axes.set_xlabel('Epochs')
    axes.set_ylabel('Loss')
    axes.set_yscale('log')
    axes.legend()
    axes.set_title('avg. MSE-loss, learning rate: ' + str(best_learning_rate) + ' training samples: ' + f'{samples:,}')

    plt.tight_layout()
    plt.savefig("output/data_driven/training/godunov_flux_" + out_str + ".png")
