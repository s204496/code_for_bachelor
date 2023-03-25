import numpy as np
import matplotlib.pyplot as plt
import math
import sys


A = np.matrix([[2, 3], [4, 1]])
ql = np.matrix([[1], [2]])
qr = np.matrix([[2], [-3]])

#make a random 2x2 matrix of integers
def Riemann_solver(q_l, q_r, A):
   D, R = np.linalg.eig(A)
   i = np.argsort(D)
   eigen_v = D[i]
   R = R[:, i]
   L = np.linalg.inv(R)
   alpha = L * (q_r - q_l)
   q_m1 = q_l + (alpha[0].item() * R[:, 0])
   ## making a check of the calculation
   q_m2 = q_r - (alpha[1].item() * R[:, 1])
   states = {
        "left": q_l.flatten(),
        "middle": q_m1.flatten(),
        "right": q_r.flatten()
    }

   waves = {
       "left": round(eigen_v[0],10),
       "right": round(eigen_v[1],10)
   }

   return states, waves

Riemann_solver(ql, qr, A)
print("The states are: " + str(states))
print("The waves are: " + str(waves))