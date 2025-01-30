import numpy as np

def rref(A):
    A = A.astype(float)  # Convert to float for division
    rows, cols = A.shape
    lead = 0  # Column index

    for r in range(rows):  # Loop through each row
        if lead >= cols:
            break
        
        # Find the first row with a nonzero value in the lead column
        i = r
        while A[i, lead] == 0:
            i += 1
            if i == rows:  # If no pivot found in column, move to next column
                i = r
                lead += 1
                if lead == cols:
                    return A
       
        # Swap current row with the row containing the pivot
        A[[r, i]] = A[[i, r]]

        # Normalize the pivot row (make leading coefficient = 1)
        A[r] /= A[r, lead]

        # Eliminate all other entries in the column
        for i in range(rows):
            if i != r:
                A[i] -= A[i, lead] * A[r]  # Row operation

        lead += 1  # Move to the next column

    return A

# Example matrix
A = np.array([[-3, 2, -1],
              [6, -6, 7],
              [3, -4, 4]])

# Compute RREF
rref_matrix = rref(A.copy())
print(rref_matrix)
