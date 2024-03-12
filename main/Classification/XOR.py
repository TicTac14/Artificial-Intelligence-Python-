import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
INPUT_SIZE = 1
DATA_SIZE = 1000
MAX_X = 10
MIN_X = -10
HIDDEN_1_NUM = 6
HIDDEN_2_NUM = 3
OUTPUT_NUM = 1
LEARN_RATE = 0.00055
x = torch.arange(MIN_X, MAX_X, (MAX_X-MIN_X)/DATA_SIZE, dtype=torch.float32).reshape((DATA_SIZE, INPUT_SIZE))

y_true = 2 * x**2


w1 = torch.randn(INPUT_SIZE, HIDDEN_1_NUM, requires_grad=True)
b1 = torch.zeros(1, HIDDEN_1_NUM, requires_grad=True)
w2 = torch.randn(HIDDEN_1_NUM, HIDDEN_2_NUM, requires_grad=True)
b2 = torch.zeros(1, HIDDEN_2_NUM, requires_grad=True)
w3 = torch.randn(HIDDEN_2_NUM, OUTPUT_NUM, requires_grad=True)
b3 = torch.zeros(1, OUTPUT_NUM, requires_grad=True)

def relu(x):
    return torch.maximum(x, torch.tensor(0.0))

def forward(x):
    hidden_1 = relu(x @ w1 + b1)
    hidden_2 = relu(hidden_1 @ w2 + b2)
    output = hidden_2 @ w3 + b3
    return output

def mean_squared_error(predicted, actual):
    return ((predicted-actual)**2).mean()


for epoch in range(100000):
    
    y_pred = forward(x)

    loss = mean_squared_error(y_pred, y_true)

    loss.backward()

    with torch.no_grad():
        w1 -= w1.grad * LEARN_RATE
        b1 -= b1.grad * LEARN_RATE
        w2 -= w2.grad * LEARN_RATE
        b2 -= b2.grad * LEARN_RATE
        w3 -= w3.grad * LEARN_RATE
        b3 -= b3.grad * LEARN_RATE

        w1.grad.zero_()
        w2.grad.zero_()
        w3.grad.zero_()
        b1.grad.zero_()
        b2.grad.zero_()
        b3.grad.zero_()

    if ((epoch + 1) % 1000 == 0):
        print(f"Epoch:{epoch +1} loss: {loss.item()}")


plt.scatter(x, y_true, color='b')
plt.plot(x, [forward(torch.tensor([y])).item() for y in x], color='r')

plt.grid()
plt.show()



