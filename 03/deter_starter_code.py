# Recursive computation of the determinant starter code. Tutorial 3 of MATH/CSCI2072 CSI, Ontario Tech U, 2024.
import numpy as np

def deter(A):
    # Recursive computation of the determinant. Input: n X n array A. Out: determinant of A (float).
    n = A.shape[0]  # Extract number of rows.
    if n == 1:      # Base case for 1x1 matrix.
        return A[0, 0]
    elif n == 2:    # For 2x2 matrix, use definition.
        return A[0, 0]*A[1, 1] - A[0, 1]*A[1, 0]
    else:           # For larger matrices, compute recursively.
        detA = 0    # Initialize the determinant.
        for i in range(n):  # Loop through each element of the first row/column.
            subA = np.delete(np.delete(A, 0, axis=0), i, axis=1)  # Create submatrix for minors.
            detA += (-1)**i * A[0, i] * deter(subA)  # Recursive calculation
        return detA
