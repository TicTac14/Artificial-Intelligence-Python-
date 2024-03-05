import matplotlib.pyplot as plt
import torch


def sigmoid(x):
    return 1/(1 + torch.exp(-x))

def sig_deriv(x):
    return torch.exp(-x)/(1 + torch.exp(-x))**2

xs = torch.arange(-20, 20, 0.1)
ys = sigmoid(xs)
deriv = sig_deriv(xs)

plt.plot(xs, ys, color='b', label='Sigmoid')
plt.plot(xs, deriv, color='r', label='Sigmoid Derivative')
plt.legend()
plt.grid()
plt.show()

def derivative_function(x):
    g = 9.81
    u = 0.15
    return 0.632*(g*torch.cos(x * torch.pi / 180)+g*u*torch.sin(x * torch.pi / 180))/(2*torch.sqrt(g*torch.sin(x * torch.pi / 180)-g*u*torch.cos(x * torch.pi / 180)))

x_vals = torch.arange(0, 180, 0.01)
y_vals = derivative_function(x=x_vals)


plt.plot(x_vals, y_vals)
plt.show()