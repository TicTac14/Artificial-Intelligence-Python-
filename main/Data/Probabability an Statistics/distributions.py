import torch
import matplotlib.pyplot as plt
import math

torch.pi = torch.acos(torch.zeros(1)) * 2

### 22.8.1 Bernoulli

p = 0.2
plt.stem([0, 1], [1-p, p])
plt.xlabel('x')
plt.ylabel('p.m.f')
plt.show()
"""
This is the probability mass function of the bernoulli distribution
in short, if x < 0, then f(x) = 0, 
if 0 <= x < 1 then f(x) = 1 - p
if x >= 1 then f(x) = 1
"""
tensor = torch.arange(-1, 2, 0.01)

def F(x):
    return 0 if x < 0 else 1 if x > 1 else 1 - p
plt.plot([-1 + i for i in range(3)], [1, 2, 3])
plt.show()

bernoulli_random_var = 1 * (torch.rand(10, 10) < p)
print(bernoulli_random_var)
# arbitrary bernoulli random variable in pytorch above


### 22.8.2 Discrete Uniform

n = 5
plt.stem([i for i in range(n)], n*[1/n])
plt.xlabel('x')
plt.ylabel('p.m.f')
plt.show()

"""
if x < 1, f(x) = 0
if k <= x < k + 1 with 1 <= k < n then f(x) = k/n
if x >= n, f(x) = n

"""

#sample an array of arbitrary sape from discrete random variable
random_discrete_uniform = torch.randint(1, n, size=(10, 10))

print(random_discrete_uniform)

### 22.8.3 Continuous Uniform

# Probability Density Function
"""
funciton definition ->

p(x) = 
        if x in [a, b] then 1/(b-a)
        if x not in [1, b] then 0 
        
"""

a, b = 1, 3
x = torch.arange(0, 4, 0.01)
p = (x > a).type(torch.float32)*(x < b).type(torch.float32)/(b-a)
plt.plot(x, p)
plt.show()

# Cumulative Distribution Function

"""
    function definition ->

p(x) = 
        if x < a, then 0
        if x in [a, b] then (x-a)/(b-a)
        if x >= b then 1
        
"""

def F(x):
    return 0 if x < a else 1 if x >= b else (x-a)/(b-a)

plt.plot(x, torch.tensor([F(y) for y in x]))
plt.xlabel('x')
plt.ylabel('c.d.f')
plt.show()

torch_tensor = (b-a) * torch.rand(10, 10) + a# torch representation of uniform randonm variable
print(torch_tensor)

## 22.8.4 Gaussian 










