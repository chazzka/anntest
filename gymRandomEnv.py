import gym

env = gym.make("BipedalWalker-v3")

"""random
"""
env.reset()
for i in range(1000):
    # action to move
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
    print(f"step{i}", obs, reward, action)
    if done:
        break

env.close()
