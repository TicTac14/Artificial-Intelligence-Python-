import pandas as pd
import torch
import numpy as np
import math
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional

file_path = '/Users/fm/Desktop/ML_Datasets/HousingData.csv'
# Get data with Pandas
data = pd.read_csv(file_path)
# Fill in Mising Values
data = data.fillna(data.mean())
# Seperate Features with the Targets
data_features = data.iloc[:, :data.shape[1]-1]
data_targets = data.iloc[:, data.shape[1]-1: data.shape[1]]
# Convert to Pytorch Tensor
data_features_tensor = torch.from_numpy(data_features.to_numpy(dtype=np.float32))
data_targets_tensor = torch.from_numpy(data_targets.to_numpy(dtype=np.float32))

## Seperate Tensor into Training, Testing, and Validation Datasets
# Seperating Training Data
train_amount = math.floor(len(data_features_tensor) * 0.70) # 70-20-10 train test validation split

train_features = data_features_tensor[:train_amount]
train_targets = data_targets_tensor[:train_amount]

#Seperating Testing Data
test_amount = train_amount + math.floor(len(data_features_tensor) * 0.20)
test_features = data_features_tensor[train_amount:test_amount]
test_targets = data_targets_tensor[train_amount:test_amount]

#Seperating Validation Data
validation_features = data_features_tensor[test_amount:]
validation_targets = data_targets_tensor[test_amount:]

# Load Data into Pytorch DataLoader Class - for mini-batch and shuffling
train_batch_size = 1
test_batch_size = 32
validation_batch_size = 16

train_dataset = TensorDataset(train_features, train_targets)
data_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

test_dataset = TensorDataset(test_features, test_targets)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

validation_dataset = TensorDataset(validation_features, validation_targets)
validation_loader = DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=False)

class Linear_Regression_Model(nn.Module):
    def __init__(self, num_inputs) -> None:
        super(Linear_Regression_Model, self).__init__()
        self.linear = nn.Linear(in_features=num_inputs, out_features=1)
        self.loss_history = []

    def forward(self, x):
        return self.linear(x)

class Multi_Layer_Perceptron_Model(nn.Module):
    def __init__(self, num_inputs, num_hidden1, num_hidden2, num_outputs) -> None:
        super(Multi_Layer_Perceptron_Model, self).__init__()
        self.linear1 = nn.Linear(num_inputs, num_hidden1)
        self.hidden1 = nn.ReLU()
        self.linear2 = nn.Linear(num_hidden1, num_hidden2)
        self.hidden2 = nn.ReLU()
        self.linear3 = nn.Linear(num_hidden2, num_outputs)
        self.loss_history = []
 
    def forward(self, x):
        hidden1_output = self.hidden1(self.linear1(x))
        hidden2_output = self.hidden2(hidden1_output)
        output = self.linear3(hidden2_output)
        return output

model = Multi_Layer_Perceptron_Model(train_features.shape[1], 16, 16, 1)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 10000

for epoch in range(epochs):
    model.train()
    for X, y_true in data_loader:

        optimizer.zero_grad()
        y_pred = model(X)

        loss = loss_function(y_pred, y_true)
        loss.backward()

        optimizer.step()

    if (epoch % 100 == 0):
        model.loss_history.append([epoch, loss.item()])
        print(f"Epoch: {epoch + 1} Loss: {loss.item()}")


plt.plot(torch.tensor(model.loss_history)[:, 0], torch.tensor(model.loss_history)[:, 1], color='b')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Epoch vs Loss')
plt.show()

def evaluate_model(model, data_loader):
    model.eval()
    total_loss = 0
    total_samples = 0


    with torch.no_grad():
        for data, targets in data_loader:
            predicted = model(data)
            loss = torch.nn.functional.mse_loss(predicted, targets)
            total_loss += loss.item() * data.size(0)
            total_samples += data.size(0)
    
    mse = total_loss / total_samples
    return mse






