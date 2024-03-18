import torch
import torchvision
import matplotlib.pyplot as plt

train_data = torchvision.datasets.MNIST(
    root = 'data',
    train = True,                         
    transform = torchvision.transforms.ToTensor(), 
    download = True,            
)
test_data = torchvision.datasets.MNIST(
    root = 'data', 
    train = False, 
    transform = torchvision.transforms.ToTensor()
)
print(train_data.data.shape) # the shape of train data
print(train_data.targets.shape)# the shape of targets for each train_data

# plt.imshow(train_data.data[2], cmap='gray')
# plt.title('%i' % train_data.targets[2])
# plt.show()

test_tensor = torch.tensor([
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ],
    [
        [10, 11, 12],
        [13, 14, 15],
        [17, 18, 19]
    ]
])
#Conversion to 2D vectors for model
two_dim_train = train_data.data.flatten(1)
print(two_dim_train.shape)

class SoftmaxClassificationModel():
    def __init__(self, num_inputs, num_outputs, lr):
        self.w = torch.randn(size=(num_inputs, num_outputs), 
                            requires_grad=True)
        self.b = torch.zeros(size=(1, num_outputs), requires_grad=True)
        self.learn_rate = lr

    def softmax(self, X):
        exp_X = torch.exp(X)
        return exp_X / exp_X.sum(1, keepdim=True)

    def categorical_loss(self, y_pred, y_true):
        return -torch.log(y_pred[list(range(len(y_pred))), y_true]).mean()

    def forward(self, X):
        return self.softmax(X @ self.w + self.b)
    
    def gradient_descent(self, X, y_true, epochs):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.categorical_loss(y_pred, y_true)
            loss.backward()

            with torch.no_grad():
                self.w -= self.learn_rate * self.w.grad
                self.b -= self.learn_rate * self.b.grad

                self.w.grad.zero_()
                self.b.grad.zero_()
            if (epoch % 50 == 0):
                print(f"Epoch:{epoch + 1} Loss: {loss.item()}")
        print('Finished')

 
X = train_data.data.flatten(1)/255 # normalize
y_true = train_data.targets

model = SoftmaxClassificationModel(X.shape[1], (y_true.max() + 1).item(), 1)
model.gradient_descent(X=X, y_true=y_true, epochs=1000)

total_correct = 0
average = 0

for i in range(test_data.data.shape[0]):
    test_tensor_x, test_tensor_y = test_data.data.flatten(1)[20]/255, test_data.targets[20]
    prediction = model.forward(test_tensor_x).flatten(0)
    average += torch.max(prediction).item() * 100
    if torch.max(prediction).item() == prediction[test_tensor_y.item()].item():
        total_correct += 1

average *= 1/((y_true.max() + 1).item())

print(f"Test Accuracy: {total_correct/test_data.data.shape[0] * 100}%")
print(f"Average Prediction Confidence: {average}")





