# README

There are multiple different files, here is a describtion of the content of each file and diretory.

## Requirements

You need the following python packages to be able to run the project
1. matplotlib:                  ```pip install matplotlib```
2. numpy:                       ```pip install numpy```
3. pandas:                      ```pip install pandas```
4. pytest                       ```pip install pytest```
5. pytest-coverage              ```pip install pytest-cov```
6. pytorch                      ```pip install torch```
7. torch-vision(GPU support)    ```pip install torchvision``` 

## Directory: exact_Riemann_solver

This directory contains all the files to solve the Shallow Water equations in 1D exactly (to machine epsilon) by using an exact Riemann solver, the code within this directory is heavily inspired by the FORTRAN implementation p. 128-140 in "Shock-Capturing Methods for Free-Surface Shallow Flows" - by Eleuterio F. Toro.

### Structure of directory:
```bash
exact_Riemann_solver/
├── aux_functions
│   ├── dry_bed.py
│   ├── file_writer.py
│   └── wet_bed.py
├── data
├── input
│   └── test1.txt
└── main.py
```

### execution

From the root the code can be executed like:

```bash
python `exact_riemann_solver/main.py test1.txt output.txt
```

Which will read from ```exact_riemann_solver/input/test1.txt``` and write to ```exact_riemann_solver/data/output.txt```

## Testing

With the pytest installed and the local path to the bin file, you should be able to run the unit tests in the following manner:

```pytest```

#### **parameters of input file**
1. adskfj
2. asdflkjheavily

### Files

#### Exact_Riemann_solver/main.py

This is the main file, which take in an input file from the directory ```exact_Riemann_solver/input```. In this file the parameters specified below should be stated:

## Unrelated other files

#### File: 1D_advection.ipynb

This file contains an exact solution to the scalar advection equation. And 3 different numerical schemes to try and solve the same problem. Two of the schemes Upwind method and Lax-Wendroff are stable, and the third numerical scheme (An unstable flux) is unstable. The 3 schemes can be found in Leveque - Finite volume methods for Hyperbolic problems (4.18, 4.23, 4.25). You can togle whether the methods are used as finite volume methods or finite difference methods. Further there are 3 possible initial conditions that can be choosen, the problems are processed as IVBP with transitive boundaries. The error of each scheme is computed by comparing the exact solution to the approximated solution from each scheme. The 2-norm is used to measure the error formular is given in: Finite Differnce Methods for Ordinary and Partial Differential Equation - by Leveque.  

The last plot in the file shows the global order of accuracy for both the Upwind method and the Lax-Wendroff method, showing that with the given norm at a specific time T, the upwind method converges at rate $O(\Delta h)$ to the exact solution and Lax-Wendroff converges with the rate $O(\Delta h^2)$.

#### File: 2x2_constant_coefficients_Riemann_solver.ipynb

This file shows how to solve the Riemann problem for the constant coefficient system $q_t+Aq_x=0$, where A is a 2 by 2 matrix that is diagonalizble. We get the two waves speed, and the state that is left in between the two waves. 

### References to Toro 
In the code comments, which are references to the book "Shock-Capturing Methods for Free-Surface Shallow Flows" by Eleuterio F. Toro - 2001. These reference will look like (Toro (x.x) - page xxx), where the x's are equation number and page number.
