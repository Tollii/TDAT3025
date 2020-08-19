import os
import torch
import matplotlib.pyplot as plt

x_train = [] # length
y_train = [] # weight

# x_train = torch.tensor([1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(-1, 1)
# y_train = torch.tensor([5.0, 3.5, 3.0, 4.0, 3.0, 1.5, 2.0]).reshape(-1, 1) 

with open('./data/length_weight.csv') as file:
    comment = file.readline()

    for line in file:
        observation = line.split(',')
        x_train.append(float(observation[0]))
        y_train.append(float(observation[1]))

list_length = len(x_train)

x_train = torch.tensor([x_train]).reshape(-1, 1)
y_train = torch.tensor([y_train]).reshape(-1, 1)

class LinearRegressionModel:

    def __init__(self):
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    def f(self, x):
        return x @ self.W + self.b

    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))

model = LinearRegressionModel()

optimizer = torch.optim.SGD([model.b, model.W], 0.01)
for epoch in range(100):
    print(f'Weight {epoch}: {model.W}')
    print(f'Bias {epoch}: {model.b}')
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

# Visualize result
plt.plot(x_train, y_train, 'o', label='$(\\hat x^{(i)},\\hat y^{(i)})$')
plt.xlabel('x')
plt.ylabel('y')
x = torch.tensor([[torch.min(x_train)], [torch.max(x_train)]])
plt.plot(x, model.f(x).detach(), label='$y = f(x) = xW+b$')
plt.legend()
plt.show()
