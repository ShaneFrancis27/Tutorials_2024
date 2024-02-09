# Speed test: computing the determinant recursively or with LU decomposition.
import numpy as np
import time                              # For timing
import matplotlib.pyplot as plt
from LU import LU
from deter_starter_code import deter

# Specify the range of matrix sizes to test
istart = 2
iend = 10

# Initialize arrays to store wall times for each matrix size
wtimerec = np.zeros((iend-istart+1, 2))  # For the recursive method
wtimeLU = np.zeros((iend-istart+1, 2))   # For the LU decomposition method

# Test recursive computation
for i, n in enumerate(range(istart, iend+1)):
    A = np.random.rand(n, n)             # Generate a random n x n matrix
    start = time.time()                  # Start timer
    detA = deter(A)                      # Calculate determinant recursively
    elapsed = time.time() - start        # Measure elapsed time
    wtimerec[i, :] = [n, elapsed]        # Store the matrix size and elapsed time

    print(f"Recursive: n={n}, wall time={elapsed:.6f} seconds, det={detA:.2e}")

# Test LU decomposition method
for i, n in enumerate(range(istart, iend+1)):
    A = np.random.rand(n, n)             # Generate a random n x n matrix
    start = time.time()                  # Start timer
    L, U, ok = LU(A)                     # Perform LU decomposition
    if ok == 1:
        detA = np.prod(np.diag(U))       # Calculate determinant from U
        elapsed = time.time() - start    # Measure elapsed time
        wtimeLU[i, :] = [n, elapsed]     # Store the matrix size and elapsed time

        print(f"LU: n={n}, wall time={elapsed:.6f} seconds, det={detA:.2e}")
    else:
        print(f"LU decomposition failed for n={n} due to near-zero pivot.")

# Plotting the results
plt.figure(figsize=(10, 6))
plt.loglog(wtimerec[:, 0], wtimerec[:, 1], '-*r', label='Recursive')
plt.loglog(wtimeLU[:, 0], wtimeLU[:, 1], '-*g', label='LU Decomposition')
plt.title('Determinant Calculation Performance')
plt.xlabel('Matrix Size (n)')
plt.ylabel('Wall Time (seconds)')
plt.legend()
plt.grid(True)
plt.show()
