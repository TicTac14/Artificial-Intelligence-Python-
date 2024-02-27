import torch

#2.3.1 Scalars

tensor_one = torch.tensor(3.0)
tensor_two = torch.tensor(2.0)

scalar_sum = tensor_one + tensor_two #simply adds scalars
scalar_mult = tensor_one * tensor_two # simply multipliies scalars
scalar_div = tensor_one/tensor_two #divides scalars
scalar_power = tensor_one**tensor_two #raises first tensor to the power of the other tenswor value
print(scalar_sum, scalar_mult, scalar_div, scalar_power)

#2.3.2 Vectors

t_x = torch.arange(3)
print(t_x)

# t_x is a vector and can be visualualized by stacking wach element ontop of one another
# this would be a 1d vector
print(t_x[2]) # return a new tensor with a scalar value equal to the value at index 2
print(len(t_x)) # returns number of elements in tensor
print(t_x.shape) # return torch.Size

A = torch.arange(6).reshape(3, 2)
print(A)

#this would result in a 2d vector because it has two dimentions, right left and up down

A_transposed = A.T # transposes matrix so flips rows and columns
                   # so for every a[i][j], a[i][j] = a[j][i]
A = torch.tensor([[1, 2, 3],
                  [2, 0, 4],
                  [3, 4, 5]])
print(A.T == A)

#2.3.4 Tensors

t = torch.arange(24).reshape((2, 3, 4))
print(t)

# 2.3.5 Basic Properties of Tensor Arithmetic
A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
B = A.clone()
print(A, A + B, A * B) # basic tensor arithmetic

a = 2
X = torch.arange(24).reshape(2, 3, 4) 
print(a + X, (a * X).shape) # tensor srithmetic with scalars

#2.3.6 Reduction

var1 = torch.arange(3, dtype=torch.float32)
print(var1, var1.sum()) # calculating sum of all vals in tensor

sum_on_specified_axis = A.sum(axis=0) # axis=0 means by column and axis=1 means by each row

print(A, sum_on_specified_axis)

mean = A.mean() # takes the mean of the tensor
print(mean)

# 2.3.7 Non-Reduction Sum

sum_A = A.sum(axis=1, keepdim=True) # this takes the sum of a tensor along an axis but keeps its dimentions
print(sum_A, sum_A.shape)

# 2.3.8 Dot Product

y = torch.ones(3, dtype=torch.float32)
x = torch.arange(3, dtype=torch.float32)
vect_vect_dot_product = torch.dot(x, y) # gets the dot product of the tensor
# this operation only works with 1D tensors
# it basically multiplies the tensors element wise and then takes the sum
vect_vect_dot_product_unsimplified = torch.sum(x * y)
print(x, y, vect_vect_dot_product, vect_vect_dot_product_unsimplified)

# 2.3.9 Matrix-Vector Products
matrix = torch.arange(6, dtype=torch.float32).reshape((2, 3))
vector = torch.arange(3, dtype=torch.float32)

matrix_vector_products = torch.mv(matrix, vector)
# this seems a bit confusing as to how it is calculating
# element wise multiply row in matrix by the vector, then sum up all the values in the row
# do this for each row and get a final 1D tensor
print(matrix, vector, matrix_vector_products)

#2.3.10 Matrix-Matrix Multiplication

matrix1 = torch.arange(6, dtype=torch.float32).reshape((2, 3))
matrix2 = torch.arange(6, dtype=torch.float32).reshape((3, 2))
print(matrix1, matrix2)
matrix_matrix_dot_product = torch.mm(matrix1, matrix2)
print(matrix_matrix_dot_product)

# each row is multiplied by each of the colums of the other matrix
# the result is an n x m matrix where n is the the amount of rows in teh first matrix
# and the m is the amount of colums on the second matrix

#2.3.11 Norms

u = torch.tensor([3.0, -4.0])
norm_u = torch.norm(u)

"""
the norm is teh length of the vector
its the square root of the sum of the squares of each index
pythagorean theorem essentially
this type of norm is called l2
Another type of norm is the sum of all the absolute values of the elements.
This is called the Manhattan distance.
"""

print(norm_u)

# Excersises

# 1.
A = torch.arange(12).reshape((2, 6))
solution = A.T.T == A
print(solution)

# 2. 
B = torch.arange(10, 22).reshape((2, 6))
solution = A.T + B.T == (A + B).T
print(solution)

#3. 
n = 2
A = torch.arange(n**2, dtype=torch.float32).reshape((n, n))
solution = A + A.T
print(solution.T == solution)

"""
Property A -> (A.T).T = A
Property B -> A.T + B.T = (A + B).T

Proof -> (A + A.T).T = A + A.T
         By property B ->
         A.T + (A.T).T = A + A.T
         By property A ->
         A.T + A = A + A.T
         This statement is true, therefore, A + A.T is always symmetric

"""

# 4. 

X = torch.arange(24, dtype=torch.float32).reshape((2, 3, 4))

vect1 = torch.tensor([1, 2, 3])
vect2 = torch.tensor([5, 1, 2])

print(torch.mm(vect1, vect2))

















