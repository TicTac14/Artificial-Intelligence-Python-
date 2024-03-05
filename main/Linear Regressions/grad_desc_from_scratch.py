import torch

xs = torch.arange(-5, 5, 0.01).view(-1, 1)
ys = xs * 25.1 + 50

w = 1.5231
b = 0.8139

def model(x):
    return w * x + b

def L(y_pred, y_true):
    return ((y_pred-y_true)**2).mean()

epochs = 10000
learning_rate = 0.1

for epoch in range(epochs):
    prediction = model(xs)
    loss = L(prediction, ys)

    w_grad = (-xs * 2 * (ys - prediction)).mean() # the partial derivative of the loss in respect to w
    b_grad = (-2 * (ys - prediction)).mean() # the partial derivative of the loss in respect to b

    w -= learning_rate * w_grad
    b -= learning_rate * b_grad


    print(f"Epoch {epoch + 1}: loss = {loss.item()}")

print(f'Finished, w = {w} b = {b}')