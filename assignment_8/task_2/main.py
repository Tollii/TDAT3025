import gym
import numpy as np
import math
from collections import deque
import torch
import torch.nn as nn
import random




class CartPole(nn.Module):
    def __init__(self, buckets=(1, 1, 6, 12,), n_episodes=1000, n_win_ticks=200, min_epsilon=0.1, batch_size=64, gamma=1.0, epsilon_decay=0.995):
        super(CartPole, self).__init__()
        self.buckets = buckets # down-scaling feature space to discrete range
        self.n_episodes = n_episodes # training episodes 
        self.n_win_ticks = n_win_ticks # Number of ticks needed to "win"
        self.min_epsilon = min_epsilon # exploration rate
        self.gamma = gamma # discount factor
        self.epsilon_decay = epsilon_decay
        self.epsilon = 1
        self.batch_size = batch_size
        self.env = gym.make('CartPole-v0')
        self.memory = deque(maxlen=10000)

        # Model
        self.model = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(256, 2)
                ).double()
        self.optimizer = torch.optim.Adam(self.model.parameters(), 0.001)

    def remember(self, *args):
        self.memory.append((*args,None))


    def replay(self, batch_size):
        x_batch, y_batch = [], []
        minibatch = random.sample(
            self.memory, min(len(self.memory), batch_size))
        for state, action, reward, next_state, done in minibatch:
            y_target = self.f(torch.from_numpy(state).unsqueeze(1).unsqueeze(1))
            y_target[0][action] = reward if done else reward + \
                self.gamma * np.max(self.f(torch.from_numpy(next_state).unsqueeze(1).unsqueeze(1)).detach().numpy()[0])
            x_batch.append(state[0])
            y_batch.append(y_target[0])

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay


    # Predictor
    def f(self, x):
        return torch.softmax(self.model(x), dim=1)

    # Cross Entropy loss
    def loss(self, x, y):
        return nn.functional.cross_entropy(self.model(x), y.argmax(1))

    # Accuracy
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)))

    # Make input discrete
    def discretize(self, obs):
        upper_bounds = [self.env.observation_space.high[0], 0.5, self.env.observation_space.high[2], math.radians(50)]
        lower_bounds = [self.env.observation_space.low[0], -0.5, self.env.observation_space.low[2], -math.radians(50)]
        ratios = [(obs[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)

    # Educated action, or random action
    def choose_action(self, state, epsilon):
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.f(torch.from_numpy(state).unsqueeze(1).unsqueeze(1)).detach().numpy)

    def get_epsilon(self, tick):
        return max(self.min_epsilon, min(self.epsilon, 1.0 - math.log10((tick + 1) * self.epsilon_decay)))

    def preprocess_state(self, state):
        return np.reshape(state, [1, 4])

    def run(self):
        scores = deque(maxlen=100)

        for e in range(self.n_episodes):
            current_state = self.preprocess_state(self.env.reset())

            done = False
            i = 0

            # One simulation
            while not done:
                # env.render()
                action = self.choose_action(current_state, self.get_epsilon(e))
                obs, reward, done, _ = self.env.step(action)
                new_state = self.preprocess_state(obs)
                self.remember(current_state, action, reward, new_state)
                current_state = new_state
                i += 1

            self.optimizer.step()
            self.optimizer.zero_grad()
            scores.append(i)
            mean_score = np.mean(scores)
            if mean_score >= self.n_win_ticks and e >= 100:
                print(f'Ran {e} episodes. Solved after {e - 100} trials. Last run ran for {scores[-1]} ticks.')
                return e - 100

            if e % 100 == 0:
                print(f'Episode {e} - Mean survival time over last 100 episodes was {mean_score} ticks.')

            self.replay(self.batch_size)

        print(f'Unable to solve after {e} episodes')
        return e

if __name__ == "__main__":
    solver = CartPole()
    solver.run()
