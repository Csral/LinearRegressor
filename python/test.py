from matrix import Matrix, LinearAlgebra

A = Matrix([[-3, 2, -1],
              [6, -6, 7],
              [3, -4, 4]])

B = Matrix([[-1, 4, -1],
              [9, -6, 7],
              [3, 4, 4]])

print("\n",A.det())

X = Matrix([[1],[2],[3]])

linalg = LinearAlgebra()

a = linalg.rref(A)
print(a)
#* Validate here
c = linalg.multiply(A,B)
print(c)

d = linalg.transpose(c)
print(d)

e = linalg.solver(A,X)
print("-"*100, "\n")
print(e)

aug = linalg.augment_matrices(A,X)

rf = linalg.rref(aug)
print(rf)