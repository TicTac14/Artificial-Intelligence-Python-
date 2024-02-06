#2.1.1 Getting Started
import torch
import numpy as np

# this creates a tensor with 12 nums equally spaced 0-included 12-not included
x = torch.arange(12, dtype=torch.float32)
print(x)

number_of_elements = x.numel() # returns number of elments in tensor
x_shape = x.shape # returns tensor.Size(shape) with given shape like [12] or [12, 12]
reshaped_x = x.reshape((3, 4)) # reshapes tensor, multiplying all should give you total elements equal to the num elements in original tensor
print(reshaped_x)

zeros_tensor = torch.zeros((2, 2, 3)) # creates tensor with all 0's
ones_tensor = torch.ones((2, 3, 4)) # creates tensor with all 1's
rand_tensor = torch.randn((3, 4)) # creates a tensor of given shape with random values
tensor = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]) # manually creates tensor

# 2.1.2 Indexing and Slicing

# Notes: array[start:stop] includes start idx but not last
test_tensor = torch.arange(12, dtype=torch.float32).reshape((3, 4))
last_row = test_tensor[-1]
sliced_tensor = test_tensor[1:3] # selects row 1 and 

print(last_row, sliced_tensor)

test_tensor[0, 0] = 18 # the [0, 0] indexing indicates to get the item at that position, then you can change it

test_tensor[:2, :] = 18 # the :2 indicates to access the first two rows, : indicates to access all columns

print(test_tensor)

# 2.1.3 Operations

test_tensor_1 = torch.arange(12, dtype=torch.float32).reshape((3, 4))
exponent = torch.exp(test_tensor_1) # I don't fully understand the purposes of this

t_x = torch.tensor([1.0, 2, 4, 8])
t_y = torch.tensor([2, 2, 2, 2])
add = t_x + t_y # adds the matricies by adding each of the indexes with ethe other
substract = t_x - t_y # subtracts the matricies by subtracting each of the indexes with the other
multiply = t_x * t_y # multiplies the matricies by multiplying each of the indexes with the other
divide = t_x / t_y # divides the matricies by dividing each of the indexes with the other
to_the_power = t_x ** t_y # raises each element of t_x to the power of the corresponding element in t_y

print(add, substract, multiply, divide, to_the_power)

new_tensor_ = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
#concatonate merges two tensors into one, just provide both tensors and axis(0 = row, 1 = col) the "dim" is the axis
concatenated_tensor_row = torch.cat((test_tensor_1, new_tensor_), dim=0)
concatenated_tensor_col = torch.cat((test_tensor_1, new_tensor_), dim=1)
print(concatenated_tensor_row, concatenated_tensor_col)


areEqual = test_tensor == new_tensor_ # for each position, if test_tensor[i, j] == new_tensor_[i, j], return new tensor
print(areEqual)

sum_ = new_tensor_.sum() # returns tensor with the sum of all values in tensor

# 2.1.4 BroadCasting
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))

add_a_b = a + b 

# a and b dont have matching shapes, but it will replicate a, b until they can be added.

#2.1.5 Saving Memory
tens_1 = torch.arange(10).reshape((2, 5))
tens_2 = torch.arange(10).reshape((2, 5))
before = id(tens_1)
tens_1 = tens_1 + tens_2
does_equal_before = id(tens_1) == before
print(does_equal_before)

# 2.1.6 Conversion to Other Python Objects

test_tensor_2 = torch.arange(5, dtype=torch.float32)
numpy_ndarray = test_tensor_2.numpy() # converts torch tensor to numpy array

t_l = torch.tensor([3.5])
to_float = float(t_l)

#2.1.8 Exercises
temp_tensor = torch.tensor([1, 2, 3])
temp_tesor_2 = torch.tensor([3, 4, 5])

greater_than = temp_tensor > temp_tesor_2 # creates a tensor such that new-tensor[i, j] = old_1[i, j]  > old_2[i, j]
less_than = temp_tensor < temp_tesor_2 # creates a tensor such that new-tensor[i, j] = old_1[i, j]  < old_2[i, j]

print(greater_than, less_than)

# Excercise 2

a = torch.arange(0, 20, dtype=torch.float32).reshape((2, 2, 5))
b = torch.arange(0, 4, dtype=torch.float32).reshape((2, 2, 1))

print(a, b)
print(a + b)
print(a - b)
print(a * b)
print(a / b)
print(a ** b)




