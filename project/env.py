import gym
env = gym.make('SpaceInvaders-v0')
observation = env.reset()
print(observation)
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())
env.close()
