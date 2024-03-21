import torch
import matplotlib.pyplot as plt
from random import randint
x = torch.arange(-10, 10, 0.001, dtype=torch.float32).reshape(-1, 1)
noise = torch.normal(0, 1, size=(x.numel(), 1))
y_true = 2.5 * x + 5.0 + noise


class LinearRegressionModel():
    def __init__(self, learn_rate):
        self.w = torch.randn(size=(1, 1), requires_grad=True)
        self.b = torch.zeros(size=(1, 1), requires_grad=True)
        self.η = learn_rate
        self.loss_history_x = []
        self.loss_history_y = []

    def forward(self, x):
        return x @ self.w + self.b
    
    def loss(self, y_true, y_pred):
        return torch.mean((y_pred-y_true)**2)

    def mini_batch_gradient_descent(self, x, y_true, batch_size, epochs):
        for epoch in range(epochs):
            rand_idx = randint(0, len(x)-batch_size)
            mini_batch_x = x[rand_idx: rand_idx + batch_size]
            mini_batch_y = y_true[rand_idx:rand_idx + batch_size]
            
            y_pred = self.forward(mini_batch_x)
            loss = self.loss(mini_batch_y, y_pred)
            loss.backward()

            with torch.no_grad():
                self.w -= self.η * self.w.grad
                self.b -= self.η * self.b.grad

                self.b.grad.zero_()
                self.w.grad.zero_()

            print(f"Epoch: {epoch} Loss: {loss.item()}")
            self.loss_history_x.append(epoch)
            self.loss_history_y.append(loss.item())

    
    def full_batch_gradient_descent(self, x, y_true, epochs):
        for epoch in range(epochs):
            y_pred = self.forward(x) # tranformed x
            loss = self.loss(y_true, y_pred)

            loss.backward()

            with torch.no_grad():
                self.w -= self.η * self.w.grad
                self.b -= self.η * self.b.grad

                self.w.grad.zero_()
                self.b.grad.zero_()
            
            print(f"Epoch: {epoch} Loss: {loss.item()}")
            self.loss_history_x.append(epoch)
            self.loss_history_y.append(loss.item())

full_model = LinearRegressionModel(0.005)
full_model.full_batch_gradient_descent(x, y_true, 2000)

mini_model = LinearRegressionModel(0.001)
batch_size = 1000
mini_model.mini_batch_gradient_descent(x, y_true, batch_size, 2000)

# visualising training 
print(f'full w:{full_model.w} b:{full_model.b}')
print(f"mini w:{mini_model.w} b:{mini_model.b}")
plt.plot(full_model.loss_history_x, full_model.loss_history_y, color='r', label='full_batch')
plt.plot(mini_model.loss_history_x, mini_model.loss_history_y, color='b', label=f'mini-batch-{batch_size}')
plt.grid()
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.gca().set_xlim([0, 2000])
plt.gca().set_ylim([0, 2])
plt.show()


    
