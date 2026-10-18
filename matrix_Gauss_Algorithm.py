import numpy as np

# Gaussian elimination with scaled partial pivoting
def solveLinearSystem(A, b):

  A = A.astype(float).copy()
  b = b.astype(float).copy()

  for k in range(len(A)):
      # Partial Pivoting
      pivot_row = k
      largest_pivot = 0

      for i, row in enumerate(A[k:], start=k):
          scale = max(abs(row[k:]))
          if scale == 0:
              continue

          if abs(row[k]) / scale > largest_pivot:
              largest_pivot = abs(row[k]) / scale
              pivot_row = i

      if A[pivot_row][k] == 0:
          raise ValueError("Matrix is singular")
    
      A[[k, pivot_row]] = A[[pivot_row, k]]
      b[k], b[pivot_row] = b[pivot_row], b[k]

      # Forward Phase
      for i, row in enumerate(A[k+1:], start=k+1):
          if abs(row[k]) != 0:
              factor = A[i][k] / A[k][k]

              for index, number in enumerate(A[i]):
                  A[i][index] = A[i][index] - A[k][index] * factor

              b[i] = b[i] - b[k] * factor

  #Backward Phase
  x = [0] * len(A)

  for i in reversed(range(len(A))):
      result = b[i]

      for j in range(i + 1, len(A)):
          result = result - A[i][j] * x[j]

      x[i] = result / A[i][i]

  return x


# Check whether a linear system is consistent
def isConsistent(A, b):
    A = A.astype(float).copy()
    b = b.astype(float).copy()

    n = len(A)
    current_row = 0

    # Forward Phase
    for k in range(n):

        #find largest pivot
        pivot_row = current_row

        for i in range(current_row, n):
            if abs(A[i][k]) > abs(A[pivot_row][k]):
                pivot_row = i

        # go to next column if is no pivot
        if A[pivot_row][k] == 0:
            continue

        # swap rows
        A[[current_row, pivot_row]] = A[[pivot_row, current_row]]
        b[current_row], b[pivot_row] = b[pivot_row], b[current_row]

        # eliminate values below pivot
        for i in range(current_row + 1, n):
            factor = A[i][k] / A[current_row][k]

            for j in range(k, n):
                A[i][j] = A[i][j] - factor * A[current_row][j]

            b[i] = b[i] - factor * b[current_row]

        current_row += 1

    # check, impossible equations
    for i in range(n):
        all_zero = True

        for j in range(n):
            if abs(A[i][j]) > 1e-10:
                all_zero = False

        if all_zero == True:
            if abs(b[i]) > 1e-10:
                return False

    return True


# Calculate food quantities to meet nutritional requirements
def solveNutrients(A, b):
  x = solveLinearSystem(A,b)
  x = np.array(x) * 10
  x = np.round(x)
  
  return x
