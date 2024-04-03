import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

train_data = MNIST(
    root = 'data',
    train = True,                         
    transform = ToTensor(), 
    download = True,            
)
test_data = MNIST(
    root = 'data', 
    train = False, 
    transform = ToTensor()
)

X = torch.tensor([
    #Image 1
    [ 
        #amount of channels
        [
            #image information(height, width) so 4x4
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 255, 255, 0, 0, 1, 1, 0, 0],
            [0, 0, 255, 255, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 255, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 255, 255, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 255, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

        ]
    ]
])
kernel_size = 3
stride = 1
W = torch.tensor([
    [
        [
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ]
    ]
])

class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride) -> None:
        super(CNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride)
    
    def forward(self, x):
        return self.conv1(x)

x = torch.randint(-5, 5, size=(1, 2, 3, 3), dtype=torch.float32) # (batchSize, channels, height, width)

# in_channels are the channels in the input so 1 in this case
# out channels are the amount of filters applied to the input

model1 = CNN(in_channels=2, out_channels=2, kernel_size=2, stride=1)

# print(f"X : {x}")
# print(f"W : {model.w}") 
print(f"Convolution Model 1 : {model1(x)}")








