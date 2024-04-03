import torch
import matplotlib.pyplot as plt


features = torch.arange(-10, 10, 0.01, dtype=torch.float32).reshape((-1, 1))
targets = 23*features + 1 + torch.normal(mean=0, std=3, size=(features.numel(), 1))

inputs = features.shape[1]
hidden = 2
outputs = 1

w1 = torch.randn(size=(1, 1))
b1 = torch.zeros(size=(1, 1))

w2 = torch.randn(size=(1, 1))
b2 = torch.zeros(size=(1, 1))

w3 = torch.randn(size=(1, 1))
b3 = torch.zeros(size=(1, 1))

w4 = torch.randn(size=(1, 1))
b4 = torch.zeros(size=(1, 1))


def MSE(y_pred, y_true):
    return torch.mean((y_true-y_pred)**2)

def forward(X):
    return (w3*torch.relu(w1*X+b1) + b3) + (w4*torch.relu(w2*X+b2) + b4)

def W1_grad_at(X):
    pass

def B1_grad_at(X):
    pass


def W2_grad_at(X):
    pass

def B2_grad_at(X):
    pass


def W3_grad_at(X):
    pass
def B3_grad_at(X):
    pass


def W4_grad_at(X):
    pass
def B4_grad_at(X):
    pass




epochs = 500
learn_rate = 0.01

for epoch in range(epochs):

    pred = forward(features)

    loss = MSE(pred, targets)

    #back propogation


    # SGD optimization


    print(F"Epoch:{epoch+1} Loss:{loss.item()}")





