import torch
import numpy as np
import matplotlib.pyplot as plt
from math import pi


a, b, c, d = 1, 1, 1, 1  # Coefficients for our quadratic equation

# Generate data
X = np.linspace(-2*pi, 2*pi, 200)
noise = np.random.normal(0, 1, (200,))
y = 2 * X + 45.1 

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32).view(-1, 1)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Weights and biases
w1 = torch.randn(1, 150, requires_grad=True)  # Input to hidden layer weights
b1 = torch.zeros(1, 150, requires_grad=True)     # Hidden layer bias
w2 = torch.randn(150, 1, requires_grad=True)  # Hidden to output layer weights
b2 = torch.zeros(1, 1, requires_grad=True)      # Output layer bias

# print(w1, b1, w2, b2)

def relu(x):
    return torch.maximum(x, torch.tensor(0.0))  # ReLU activation function

def forward(x):
    hidden = relu(x @ w1 + b1)
    output = hidden @ w2 + b2
    return output

def mse_loss(predictions, targets):
    return ((predictions - targets) ** 2).mean()


# Hyperparameters
learning_rate = 0.0001
epochs = 200000

for epoch in range(epochs):
    # Forward pass
    predictions = forward(X_tensor)
    loss = mse_loss(predictions, y_tensor)

    # Backward pass
    loss.backward()  # Compute gradients

    # Update weights and biases
    with torch.no_grad():  # Temporarily disable gradient tracking
        w1 -= learning_rate * w1.grad
        b1 -= learning_rate * b1.grad
        w2 -= learning_rate * w2.grad
        b2 -= learning_rate * b2.grad

        # Manually zero the gradients after updating weights
        w1.grad.zero_()
        b1.grad.zero_()
        w2.grad.zero_()
        b2.grad.zero_()

    if (epoch+1) % 10000 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


plt.scatter(X, y, label='y_true', color='b')
y_pred = [forward(torch.tensor([y], dtype=torch.float32)).item() for y in X]

plt.plot(X, y_pred, label='y_pred', color='r')

plt.grid()
plt.legend()
plt.show()