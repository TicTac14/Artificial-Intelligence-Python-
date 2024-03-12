import torch
import random
import numpy as np
import matplotlib.pyplot as plt

xs = torch.arange(-5, 5, 0.01)
noise = np.random.normal(0, 1, (xs.numel(),))
ys = 2 * xs **2 + 2 * xs + 2 + noise

plt.scatter([x for x in xs], [y for y in ys])
plt.grid()
plt.show()



a, b, c = random.random()*10, random.random()*10, random.random()*10

def quadratic_model(x):
    return a * x**2 + b * x + c

def MSE(y_pred, y_actual):
    #mean squared error
    return ((y_actual-y_pred)**2).mean()


epochs = 20000
learning_rate = 0.005

for epoch in range(epochs):

    model = quadratic_model(xs)
    loss = MSE(model, ys)
    a_grad = 2 * (-xs**2 * (ys - model)).mean()
    b_grad = 2 * (-xs * (ys - model)).mean()
    c_grad = 2 * (-1 * (ys - model)).mean()

    a -= a_grad * learning_rate
    b -= b_grad * learning_rate
    c -= c_grad * learning_rate

    print(f"Epoch {epoch + 1}: loss = {loss.item()}")

print("finished", a, b, c)

plt.plot([x for x in xs], [y for y in quadratic_model(xs)], label='predicted', color='r')
plt.scatter([x for x in xs], [y for y in ys], label='true')

plt.legend()
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Machine Learning Model Prediction')

plt.show()
