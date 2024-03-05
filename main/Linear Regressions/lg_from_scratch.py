import torch

train_x = torch.arange(-20, 20, 0.1).view(-1, 1)  # 100 data points
train_y = 51.5 * train_x + 0.2525

w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)



def predict(x):
    return w*x + b

def getLoss(y_predicted, y_actual):
    return ((y_predicted - y_actual)**2).mean()

#optimal learning rate is around 0.000025

learning_rate = 0.005
epochs = 1000

for epoch in range(epochs):
    
    curr_model_prediction = predict(train_x)

    loss = getLoss(curr_model_prediction, train_y)

    loss.backward()
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        
        
    w.grad.zero_()
    b.grad.zero_()

    print(f"Epoch {epoch + 1}: loss = {loss.item()}")

print('finished', w, b)