import torch
import torchvision
import matplotlib.pyplot as plt


class MNIST_Model:
    def __init__(self):
        self.W = torch.rand((784, 10), requires_grad=True)
        self.b = torch.rand((1, 10), requires_grad=True)

    def logits(self, x):
        return x @ self.W + self.b

    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):
        return torch.nn.functional.cross_entropy(self.logits(x), y.argmax(1))

    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())


# Load observations from the mnist dataset. The observations are divided into a training set and a test set
mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True)
x_train = mnist_train.data.reshape(-1, 784).float()  # Reshape input
y_train = torch.zeros((mnist_train.targets.shape[0], 10))  # Create output tensor
y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1  # Populate output

mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True)
x_test = mnist_test.data.reshape(-1, 784).float()  # Reshape input
y_test = torch.zeros((mnist_test.targets.shape[0], 10))  # Create output tensor
y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1  # Populate output

model = MNIST_Model()
learning_rate = 0.5
optimizer = torch.optim.SGD([model.b, model.W], learning_rate, momentum=0.5)

accuracy_goal = 0.9

print('Training model...')
while (model.accuracy(x_test, y_test).item() < accuracy_goal):
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

print('Finished training.')

# Print model variables and loss
loss = model.loss(x_test, y_test).item()
accuracy = model.accuracy(x_test, y_test).item()
print(f'Loss = {loss}\nAccuracy = {accuracy}\n')

# Saving image of W for every number
for i in range(10):
    plt.imsave('./images/mnist/%i.png' % i, model.W[:, i].reshape(28, 28).detach())
