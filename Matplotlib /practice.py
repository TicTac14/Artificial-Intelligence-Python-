import matplotlib.pyplot as plt
import torch
import numpy as np


# def sigmoid(x):
#     return 1/(1 + torch.exp(-x))

# def sig_deriv(x):
#     return torch.exp(-x)/(1 + torch.exp(-x))**2

# xs = torch.arange(-20, 20, 0.1)
# ys = sigmoid(xs)
# deriv = sig_deriv(xs)

# plt.plot(xs, ys, color='b', label='Sigmoid')
# plt.plot(xs, deriv, color='r', label='Sigmoid Derivative')
# plt.legend()
# plt.grid()
# plt.show()

# def derivative_function(x):
#     g = 9.81
#     u = 0.15
#     return 0.632*(g*torch.cos(x * torch.pi / 180)+g*u*torch.sin(x * torch.pi / 180))/(2*torch.sqrt(g*torch.sin(x * torch.pi / 180)-g*u*torch.cos(x * torch.pi / 180)))

# x_vals = torch.arange(0, 90, 0.01)
# y_vals = derivative_function(x=x_vals)
# y_vals_2 = 0.632 * torch.sqrt(9.81*torch.sin(x_vals * torch.pi / 180)-9.81*0.15*torch.cos(x_vals * torch.pi / 180))


# plt.plot(x_vals, y_vals, color='r', label='Derivative')
# plt.plot(x_vals, y_vals_2, color='b', label='Velocity')
# plt.xlabel('Inclination(θ)')
# plt.ylabel('Velocity (m/s)')
# plt.title('Velocity-Derivative Equation')
# plt.grid()
# plt.legend()
# #Setting Limits to x and y ranges
# plt.gca().set_xlim([0, 90])
# plt.gca().set_ylim([0, 10])

# plt.show()

data_x = torch.arange(0, 50, 0.1)
data_y = data_x/25.8

plt.plot(data_x, data_y)
plt.title('Voltage vs Current')
plt.xlabel('Voltage(v)')
plt.ylabel('Current(A)')
plt.show()