import sys
sys.path.append('../SWE')
import time
import numpy as np
from aux_functions import file_manipulation, error_calculation, sampler, exact_riemann_solver
from data_driven.aux_function import general_aux, godunov_flux_aux
from numerical_schemes import godunov_upwind

def god_flux(h_l, u_l, h_r, u_r, g):
    (_, (_, _, _), boundary_flux, _) = exact_riemann_solver.solve(0.0, np.array([h_l,u_l, 0.0]), np.array([h_r,u_r, 0.0]), g, 10E-8, 50)
    return boundary_flux[0], boundary_flux[1]

def speed_test(g):
    model = general_aux.load_model('data_driven/models/godunov_flux_exact_1d6M.pt', 'cpu', 'godunov_flux') # CPU can be changed if one has a Nvidia GPU
    number_of_runs_in_test = 10000000
    data_input_l = np.zeros((number_of_runs_in_test, 2))
    data_input_r = np.zeros((number_of_runs_in_test, 2))
    for i in range(number_of_runs_in_test):
        data_input_l[i][0] = np.random.uniform(0.01, 3)
        data_input_l[i][1] = np.random.uniform(-6, 6)
        data_input_r[i][0] = np.random.uniform(0.01, 3)
        data_input_r[i][1] = np.random.uniform(-6, 6)
    # start timer
    start = time.time()
    for i in range(len(data_input_l)):
        # run riemann solver
        _, _ = god_flux(data_input_l[i][0], data_input_l[i][1], data_input_r[i][0], data_input_r[i][1], g)
        # end timer
    end = time.time()
    scheme_time = end - start
    print("calculation flux with numerical scheme: ", str(scheme_time))
    start = time.time()
    _ = godunov_flux_aux.compute(model, data_input_l[:,:], data_input_r[:,:])
    end = time.time()
    print("FFNN time: ", end - start)
    print("Speedup: ", scheme_time/(end - start))

def accuracy_test(g):
    try:
        read_file = open('input/test1.txt')
    except:
        sys.exit(1)
    (x_len, break_pos, g, cells, tolerance, iterations, t_end, h_l, u_l, _, h_r, u_r, _) = file_manipulation.extract(read_file) 
    W_l, W_r = np.array([h_l, u_l, 0.0]), np.array([h_r, u_r, 0.0])
    (_,w_exact_solver) = godunov_upwind.entire_domain("", "", False, x_len, break_pos, g, cells, 1, "", tolerance, iterations, t_end, W_l, W_r)
    (_,w_data_driven_solver) = godunov_upwind.entire_domain("", "", False, x_len, break_pos, g, cells, 3, "", tolerance, iterations, t_end, W_l, W_r)
    exact = sampler.sample_exact(break_pos, x_len, t_end, cells, g, W_l, W_r, tolerance, iterations)
    error_exact_solver = []
    error_data_driven = []
    for i in range(2):
        error_exact_solver.append(error_calculation.norm_2_fvm(exact[:, i], w_exact_solver[1:-1, i], cells, x_len))
        error_data_driven.append(error_calculation.norm_2_fvm(exact[:, i], w_data_driven_solver[1:-1, i], cells, x_len))
    print("Error exact solver: ", error_exact_solver)
    print("Error data driven solver: ", error_data_driven)
    print("Factor with which the exact solver was better for h: ", error_data_driven[0]/error_exact_solver[0])
    print("Factor with which the exact solver was better for u: ", error_data_driven[1]/error_exact_solver[1])

    # Error given same computation time, 9 times the number of cells by 80 times faster computation time
    (_,w_data_driven_solver) = godunov_upwind.entire_domain("", "", False, x_len, break_pos, g, 4500, 3, "", tolerance, iterations, t_end, W_l, W_r)
    exact = sampler.sample_exact(break_pos, x_len, t_end, 4500, g, W_l, W_r, tolerance, iterations)
    error_data_driven = []
    for i in range(2):
        error_data_driven.append(error_calculation.norm_2_fvm(exact[:, i], w_data_driven_solver[1:-1, i], 4500, x_len))
    print("Error data driven solver same time: ", error_data_driven)
    print("Factor with which the data-driven solver was better for h same time: ", error_exact_solver[0]/error_data_driven[0])
    print("Factor with which the data-driven solver  solver was better for u same time: ", error_exact_solver[1]/error_data_driven[1])


def main():
    g = 9.8
   # speed_test(g)
    accuracy_test(g)

if __name__ == '__main__':
    main()
    print("Done")
