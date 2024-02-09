# By L. van Veen, Ontario Tech U, 2024.
# LU decomposition without pivoting.
# Input: n X n matrix A.
# Output: n X n matrices L and U such that LU=A and L and U are triangular.
import numpy as np

def LU(A):
    ok = 1                                # Without pivoting, the decomposition might fail so we use a warning flag.
    small = 1e-12                         # If a pivot is smaller than this in absolute value, raise the flag.
    n = np.shape(A)[0]                    # Extract the number of rows and columns.
    U = np.copy(A)                        # Copy A into U making sure they are separate variables.                    
    L = np.identity(n)                    # Initialize L as identity matrix.
    for j in range(n):                    # Loop over columns.
        for i in range(j+1, n):           # Loop over elements below the pivot.
            if abs(U[j,j]) < small:       # Raise error flag and exit if the pivot is too small.
                print("Near-zero pivot!")
                ok = 0
                break
            L[i,j] = U[i,j] / U[j,j]     # Compute the multiplier.
            for k in range(j, n):         # Gauss elimination.
                U[i,k] = U[i,k] - L[i,j] * U[j,k]
    return L, U, ok
