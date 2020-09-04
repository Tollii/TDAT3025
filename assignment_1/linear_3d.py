import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class LinearRegressionModel:

    def __init__(self):
        self.W = torch.tensor([[0.0], [0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)
        self.state = {}

    def f(self, x):
        return x @ self.W + self.b

    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x), y)


# Read data x = length, weight. y = day
data = pd.read_csv('./data/day_length_weight.csv')
x_train = [data['length'].tolist(), data['weight'].tolist()]
y_train = data['day'].tolist()


# Transform training data
x_train = torch.tensor(x_train, dtype=torch.float)
y_train = torch.tensor(y_train, dtype=torch.float).reshape(-1, 1)


# Init model instance and optimizer
model = LinearRegressionModel()
optimizer = torch.optim.SGD([model.b, model.W], 0.0001)

print('Model training in progress... Please wait.')

for epoch in range(2000000):
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()


print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))


model.state = { 'bias': model.b, 'weight': model.W }
torch.save(model.state, './models/linear_3d.model')


# Visualize result
fig = plt.figure('3D Linear regression')
ax = fig.add_subplot(111, projection='3d')

# Plot data
ax.scatter(data['length'].tolist(), data['weight'].tolist(), data['day'].tolist(), c='blue')

# Regression line
ax.scatter(data['length'].tolist(), data['weight'].tolist(), model.f(x_train).detach(), c='red', label='stuff')

plt.legend()
plt.show()
