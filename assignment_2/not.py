import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, art3d

matplotlib.rcParams.update({'font.size': 11})

# regarding the notations, see http://stats.stackexchange.com/questions/193908/in-machine-learning-why-are-superscripts-used-instead-of-subscripts

class NOT_Model:
    def __init__(self):
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    def logits(self, x):
        return x @ self.W + self.b

    # Predictor
    def f(self, x):
        return torch.sigmoid(self.logits(x))

    def loss(self, x, y):
        # print(f'X {x}\n Y {y}')
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(x), y)


model = NOT_Model()

# Observed/training input and output
x_train = torch.tensor([[0.], [1.]])
y_train = torch.tensor([[1.], [0.]])

epochs = 200000
learning_rate = 0.001
optimizer = torch.optim.SGD([model.b, model.W], learning_rate)

print("Training model... Please wait")
for _ in range(epochs):
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()


# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

# Test NOT function
print("Not of 1 is = " + str(torch.round(model.f(torch.tensor([1.])).detach()).item()))  # prints out 0.0

fig = plt.figure("NOT operator")

plot1 = fig.add_subplot()

# Plot data points
plt.plot(x_train.detach(), y_train.detach(), 'o', label='$(\\hat x^{(i)},\\hat y^{(i)})$')

# set x and y labels
plt.xlabel('x')
plt.ylabel('y')

out = torch.reshape(torch.tensor(np.linspace(0, 1, 100).tolist()), (-1, 1))

plot1.set_xticks([0, 1])  # x range from 0 to 1
plot1.set_yticks([0, 1])  # y range from 0 to 1

x, indices = torch.sort(out, 0)

# Plot sigmoid regression curve.
plt.plot(x, model.f(x).detach())

plt.show()

