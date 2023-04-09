import numpy as np
import coefficientfind as coe

m=6
# ueven distribution example 
# x = np.array([0,0.1,0.2,0.3,0.5,1,2,3])
# even distribution example
x = np.linspace(0,7,m+2)
# initialize using sparse storage
A = np.zeros((m+2, m+2))

# first row for Neumann BC, approximates u’(x(1))
A[0,0:3] = coe.fdcoeffV(1, x[0], x[0:3])

# interior rows approximate u’’(x(i))
for i in range(1, m+1):
   A[i,i-1:i+2,] = coe.fdcoeffV(2, x[i], x[i-1:i+2])

# last row for Dirichlet BC, approximates u(x(m+2))
A[m+1,m:m+3] = coe.fdcoeffV(0, x[m+1], x[m:m+3])

print(A)