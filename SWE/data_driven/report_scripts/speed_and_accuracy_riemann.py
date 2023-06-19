import sys
sys.path.append('../SWE')
import time
import numpy as np
from aux_functions import wet_bed, file_manipulation, error_calculation, sampler
from data_driven.aux_function import general_aux, riemann_aux
from numerical_schemes import godunov_upwind

def speed_test(g):
    model = general_aux.load_model('data_driven/models/riemann_FFNN_shallow.pt', 'cpu', 'ffnn_riemann') # CPU can be changed if one has a Nvidia GPU
    number_of_runs_in_test = 10000000
    data_input_l = np.zeros((number_of_runs_in_test, 2))
    data_input_r = np.zeros((number_of_runs_in_test, 2))
    for i in range(number_of_runs_in_test):
        data_input_l[i][0] = np.random.uniform(0.01, 3)
        data_input_l[i][1] = np.random.uniform(-6, 6)
        data_input_r[i][0] = np.random.uniform(0.01, 3)
        data_input_r[i][1] = np.random.uniform(-6, 6)
    a_l_r = np.zeros((number_of_runs_in_test, 2))
    a_l_r = np.array([np.sqrt(data_input_l[:,0]*g), np.sqrt(data_input_r[:,0]*g)]).T
    mask_dry = 2*(a_l_r[:,0] + a_l_r[:,1]) > (data_input_r[:,1] - data_input_l[:,1])
    data_input_l = data_input_l[mask_dry,:] 
    data_input_r = data_input_r[mask_dry,:]
    a_l_r = a_l_r[mask_dry,:]
    # start timer
    riemann_time = 0.0
    start = time.time()
    for i in range(len(data_input_l)):
        # run riemann solver
        wet_bed.calculate(g, 10E-7, 50, data_input_l[i,0], data_input_l[i,1], a_l_r[i,0], data_input_r[i,0], data_input_r[i,1], a_l_r[i,1])
        # end timer
    end = time.time()
    riemann_time += end - start
    print("Riemann time: ", riemann_time)
    start = time.time()
    hat_hs = riemann_aux.compute(model, data_input_l[:,:], data_input_r[:,:], 2)
    end = time.time()
    print("FFNN time: ", end - start)
    print("Speedup: ", riemann_time/(end - start))

def accuracy_test(g):
    try:
        read_file = open('input/test1.txt')
    except:
        sys.exit(1)
    (x_len, break_pos, g, cells, tolerance, iterations, t_end, h_l, u_l, psi_l, h_r, u_r, psi_r) = file_manipulation.extract(read_file) 
    W_l, W_r = np.array([h_l, u_l, 0.0]), np.array([h_r, u_r, 0.0])
    (_,w_exact_solver) = godunov_upwind.entire_domain("", "", False, x_len, break_pos, g, cells, 1, "", tolerance, iterations, t_end, W_l, W_r)
    (_,w_data_driven_solver) = godunov_upwind.entire_domain("", "", False, x_len, break_pos, g, cells, 2, "", tolerance, iterations, t_end, W_l, W_r)
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

    # Error given same computation time
    (_,w_data_driven_solver) = godunov_upwind.entire_domain("", "", False, x_len, break_pos, g, 6000, 2, "", tolerance, iterations, t_end, W_l, W_r)
    exact = sampler.sample_exact(break_pos, x_len, t_end, 6000, g, W_l, W_r, tolerance, iterations)
    error_data_driven = []
    for i in range(2):
        error_data_driven.append(error_calculation.norm_2_fvm(exact[:, i], w_data_driven_solver[1:-1, i], 6000, x_len))
    print("Error data driven solver same time: ", error_data_driven)
    print("Factor with which the data-driven solver was better for h same time: ", error_exact_solver[0]/error_data_driven[0])
    print("Factor with which the data-driven solver  solver was better for u same time: ", error_exact_solver[1]/error_data_driven[1])


def main():
    g = 9.8
    #speed_test(g)
    accuracy_test(g)

if __name__ == '__main__':
    main()
    print("Done")
