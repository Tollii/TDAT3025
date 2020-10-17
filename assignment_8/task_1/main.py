import gym
import numpy as np
import math
from collections import deque

class CartPole():
    def __init__(self, buckets=(1, 1, 6, 12,), n_episodes=1000, n_win_ticks=200, min_alpha=0.1, min_epsilon=0.1, gamma=1.0, decay=25):
        self.buckets = buckets # down-scaling feature space to discrete range
        self.n_episodes = n_episodes # training episodes 
        self.n_win_ticks = n_win_ticks # Number of ticks needed to "win"
        self.min_alpha = min_alpha # learning rate
        self.min_epsilon = min_epsilon # exploration rate
        self.gamma = gamma # discount factor
        self.decay = decay
        self.env = gym.make('CartPole-v0')
        self.Q = np.zeros(self.buckets + (self.env.action_space.n,))

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
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.Q[state])

    def update_q(self, state_old, action, reward, state_new, alpha):
        self.Q[state_old][action] += alpha * (reward + self.gamma * np.max(self.Q[state_new]) - self.Q[state_old][action])

    def get_epsilon(self, tick):
        return max(self.min_epsilon, min(1, 1.0 - math.log10((tick + 1) / self.decay)))

    def get_alpha(self, tick):
        return max(self.min_alpha, min(1.0, 1.0 - math.log10((tick + 1) / self.decay)))

    def run(self):
        scores = deque(maxlen=100)

        for e in range(self.n_episodes):
            current_state = self.discretize(self.env.reset())

            alpha = self.get_alpha(e)
            epsilon = self.get_epsilon(e)
            done = False
            i = 0

            # One simulation
            while not done:
                # env.render()
                action = self.choose_action(current_state, epsilon)
                obs, reward, done, _ = self.env.step(action)
                new_state = self.discretize(obs)
                self.update_q(current_state, action, reward, new_state, alpha)
                current_state = new_state
                i += 1

            scores.append(i)
            mean_score = np.mean(scores)
            if mean_score >= self.n_win_ticks and e >= 100:
                print(f'Ran {e} episodes. Solved after {e - 100} trials. Last run ran for {scores[-1]} ticks.')
                return e - 100

            if e % 100 == 0:
                print(f'Episode {e} - Mean survival time over last 100 episodes was {mean_score} ticks.')

        print(f'Unable to solve after {e} episodes')
        return e

if __name__ == "__main__":
    solver = CartPole()
    solver.run()
