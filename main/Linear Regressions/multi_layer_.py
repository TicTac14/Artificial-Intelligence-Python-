import torch
import matplotlib.pyplot as plt
import numpy as np
import random

W_TRUE = 13.5
B_TRUE= 21.3
RANDOMNESS = 150
MIN_X = 0
MAX_X = 50
DATA_SIZE = 500
EPOCHS = 1000
LEARNING_RATE = 0.001

# Generate dataset tensors
xs = torch.arange(MIN_X, MAX_X, (MAX_X-MIN_X)/DATA_SIZE, dtype=torch.float32)

noise = np.random.normal(0, RANDOMNESS, (xs.numel(),))

ys = W_TRUE * xs + B_TRUE + noise

# Visualize Dataset
plt.scatter([x for x in xs], [y for y in ys])
plt.title('x vs y')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Define Model Parameters

w_1 = random.random() * 10
w_2 = random.random() * 10
b_1 = random.random() * 10
b_2 = random.random() * 10

# Define loss, model, sigmoid

def sigmoid(x):
    return 1/( 1 + torch.exp(-x)) # takes every value and raises e to the power of that number

def forward_(x):
    return w_2 * sigmoid(w_1 * x + b_1) + b_2

def MSE(y_actual, y_pred):
    return ((y_actual-y_pred)**2).mean()


# Gradient Descent Algorithm
for epoch in range(EPOCHS):

    # forward propogate / get current model's prediction
    prediction = forward_(xs)
    # get the cost of the prediction
    loss = MSE(ys, prediction)
    # calculate gradients in respect to each parameter


    # update parameters by going the opposite direction of the gradient mult by learn rate
    


