import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt

# Set minimal and maximal matrix size
smin = 2
smax = 20
rrs = np.ones(smax - smin + 1)  # Pre-allocate arrays for relative residual,
res = np.ones(smax - smin + 1)  # relative error and
mes = np.ones(smax - smin + 1)  # maximal error

# Loop over matrix sizes
for n in range(smin, smax + 1):
    xe = np.zeros(n)   #Exact solution
    xe[0] = 1
    A = np.zeros((n, n))  # Allocate and fill A
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            A[i - 1, j - 1] = (-1) ** (i + j) / (i + 2 * j)

    #print(A)
    r = A[:, 0]  # Set right-hand side
    print(r)
     
    x = np.linalg.solve(A, r)  # Solve by LUP decomposition

    # Compute relative residual
    residual = np.linalg.norm(np.dot(A, x) - r, 2) / np.linalg.norm(r,2)
    rrs[n - smin] = residual
    # Compute relative error
    error = np.linalg.norm((x - xe),2) / np.linalg.norm(xe,2)
    res[n - smin] = error
    # Assuming mes is meant to represent the condition number here
    mes[n - smin] = np.linalg.cond(A)

# Plotting
plt.semilogy(range(smin, smax + 1), res, '-b', label='Relative Error')
plt.semilogy(range(smin, smax + 1), mes, '-r', label='Condition Number')
plt.xlim([smin, smax])
plt.xlabel('Matrix size')
plt.ylabel('(Maximal) Relative Error')
plt.title('Maximal (red) and actual (blue) relative error')
plt.legend()
plt.show() 