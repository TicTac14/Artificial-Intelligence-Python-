import torch

x = torch.arange(4, dtype=torch.float32)
x.requires_grad_(True)
print(x.grad)

y = 2 * torch.dot(x, x)

y.backward()
print(x.grad)

x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)


