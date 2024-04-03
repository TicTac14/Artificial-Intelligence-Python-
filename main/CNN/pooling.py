import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a simple CNN architecture with one convolutional layer followed by a max pooling layer
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layer (expects 1 input channel, produces 16 feature maps, with a 3x3 kernel)
        self.conv1 = nn.Conv2d(1, 16, 3, 1)
        # Max pooling layer (2x2 window, stride 2 reduces each dimension by half)
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        # Apply convolutional layer
        x = F.relu(self.conv1(x))
        # Apply max pooling layer
        x = self.pool(x)
        return x

# Instantiate the model
model = SimpleCNN()

# Create a random tensor to represent a batch of 3 grayscale images of size 28x28
input_images = torch.rand(3, 1, 28, 28)

# Pass the input through the model
output = model(input_images)

print("Input shape:", input_images.shape)
print("Output shape after convolution and max pooling:", output.shape)
