import torch.nn  as nn
import torch

class Agent(nn.Module):

    def __init__(self):
        super(ConvolutinalNeuralNetworkmodel, self).__init__()

        self.logits = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(50176, 10)
                )

        # Predictor
        def f():
            print('wow')

        def loss():
            print('wow')

        def accuracy():
            print('wow')
