import torch
import numpy as np
import matplotlib.pyplot as plt


class NAND_Model:
    def __init__(self):
        self.W = torch.tensor([[-1.0], [1.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    def logits(self, x):
        return x @ self.W + self.b

    # Predictor
    def f(self, x):
        return torch.sigmoid(self.logits(x))

    def loss(self, x, y):
        # print(f'X {x}\n Y {y}')
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(x), y)


model = NAND_Model()

# Observed/training input and output
x_train = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
y_train = torch.tensor([[1.], [1.], [1.], [0.]])

epochs = 2000000
learning_rate = 0.0001
optimizer = torch.optim.SGD([model.b, model.W], learning_rate)

print('Training model... Please wait')
for _ in range(epochs):
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

torch.save(model.state, './models/nand')


# Print model variables and loss
print('W = %s, b = %s, loss = %s' % (model.W, model.b, model.loss(x_train, y_train)))

fig = plt.figure('NAND operator')

plot1 = fig.add_subplot(111, projection='3d')

plot1.plot(
        x_train[:, 0].squeeze().detach(),
        x_train[:, 1].squeeze().detach(),
        y_train[:, 0].squeeze().detach(),
        'o',
        label='$(\\hat x_1^{(i)}, \\hat x_2^{(i)},\\hat y^{(i)})$',
        color='blue'
        )

x1_grid, x2_grid = torch.tensor(np.meshgrid(np.linspace(-0.25, 1.25, 10),
                                            np.linspace(-0.25, 1.25, 10)), dtype=torch.float)
y_grid = torch.tensor(np.empty([10, 10]), dtype=torch.float)
for i in range(0, x1_grid.shape[0]):
    for j in range(0, x1_grid.shape[1]):
        y_grid[i, j] = model.f(torch.tensor([[x1_grid[i, j], x2_grid[i, j]]])).detach()
plot1_f = plot1.plot_wireframe(x1_grid, x2_grid, y_grid, color="green")

plt.show()
