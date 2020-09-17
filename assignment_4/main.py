import torch
import torch.nn as nn
import numpy as np


class Emojis:
   def __init__(self, data):
        self.data = np.array(data)
        self.unique_characters = set(list(''.join(self.data[:, 0])))
        self.character_to_index = dict(zip(self.unique_characters, {num for num in range(len(self.unique_characters))}))

   def encode_character(self, char):
       return np.identity(len(self.unique_characters))[self.character_to_index[char]]

   def encode_words(self):
        return [[self.encode_character(char) for char in word] for word in self.data[:, 0]]

   def encoding_to_word(self, row):
        return self.data[row.argmax(), 1]


class EmojiModel(nn.Module):
    def __init__(self, encoding_size):
        super(EmojiModel, self).__init__()

        self.lstm = nn.LSTM(encoding_size, 128)  # 128 is the state size
        self.dense = nn.Linear(128, encoding_size)  # 128 is the state size

    def reset(self, batch_size):  # Reset states prior to new input sequence
        zero_state = torch.zeros(1, batch_size, 128)  # Shape: (number of layers, batch size, state size)
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self, x):  # x shape: (sequence length, batch size, encoding size)
        out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        return self.dense(out.reshape(-1, 128))

    def f(self, x):  # x shape: (sequence length, batch size, encoding size)
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):  # x shape: (sequence length, batch size, encoding size), y shape: (sequence length, encoding size)
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))


data = Emojis([
    ['hat', '🎩'],
    ['rat', '🐀'],
    ['man', '👨'],
    ['poo', '💩'],
    ])

x_train = torch.tensor(data.encode_words(), dtype=torch.float).transpose(0, 1)
y_train = torch.tensor(np.identity(np.shape(x_train)[1]), dtype=torch.float)


print(x_train)
print(y_train)

model = EmojiModel(len(data.unique_characters))

def generate(label):
    model.reset(1)
    encoding = [data.encode_character(char) for char in label]
    y = model.f(torch.tensor(encoding, dtype=torch.float).detach().reshape(4, 1, -1))
    return dataset.encoding_to_word(y)


learning_rate = 0.001
epochs = 500
optimizer = torch.optim.RMSprop(model.parameters(), learning_rate)

for epoch in range(epochs):
    model.reset(x_train.size(1))
    loss = model.loss(x_train, y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 10 == 9:
        print(generate('poo'))
