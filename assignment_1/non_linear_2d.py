import torch
import matplotlib.pyplot as plt
import pandas as pd

class LinearRegressionModel:

    def __init__(self):
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)
        self.state = {}

    def f(self, x):
        return 20 * torch.sigmoid((x @ self.W) + self.b) + 31

    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))

# Fetch data
data = pd.read_csv('./data/day_head_circumference.csv')
x_train = data['day'].tolist()
y_train = data['head circumference'].tolist()

# Format data
x_train = torch.tensor(x_train, dtype=torch.float).t().reshape(-1, 1)
y_train = torch.tensor(y_train, dtype=torch.float).t().reshape(-1, 1)

# Init model and optimizer
model = LinearRegressionModel()
optimizer = torch.optim.SGD([model.b, model.W], 0.000001)

print('Training model... Please wait.')
for epoch in range(2000000):
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

# Save model
model.state = { 'bias': model.b, 'weight': model.W }
torch.save(model.state, './models/non_linear_2d.model')

# Print weight, bias and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

# Visualize result
plt.plot(x_train, y_train, 'o', label='$(\\hat x^{(i)},\\hat y^{(i)})$')
plt.xlabel('x')
plt.ylabel('y')
x, indices = torch.sort(x_train, 0)
plt.plot(x, model.f(x).detach(), label='$y = f(x) = 20sig(xW+b$) + 31')
plt.legend()
plt.show()
