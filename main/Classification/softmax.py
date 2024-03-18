import torch
from d2l import torch as d2l
from torchvision import datasets
from torchvision.transforms import ToTensor

def softmax(X):
    exp_X = torch.exp(X)
    return exp_X/(exp_X.sum(1, keepdims=True))

def cross_entropy(y_pred, y_true):
    return -torch.log(y_pred[list(range(len(y_pred))), y_true]).mean()

a = torch.tensor([
    [0.5, 0.5, 0],
    [0.25, 0.25, 0.5]
])
b = torch.tensor([
    [0, 1, 0],
    [1, 0, 0]
])
print(a * b)

y_pred = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_true = torch.tensor([0, 2])
num_classes = torch.max(y_true) + 1
print(num_classes)
print(cross_entropy(y_pred, y_true))

X = torch.tensor([
    [0, 5, 3, 7, 10],
])
print(softmax(X))
print(softmax(X).sum(1))

#softmax regression from scratch
input_size = 4
output_size = 3


W = torch.normal(0, 0.01, size=(input_size, output_size), requires_grad=True)
b = torch.zeros(size=(output_size, 1), requires_grad=True)

def forward(X):
    X = X.reshape((-1, W.shape[0]))
    return softmax(X @ W + b)

