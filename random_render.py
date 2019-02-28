import gym
import math
import random
import time
import numpy as np

env = gym.make("CartPole-v1")
ram_dimensions = env.observation_space.shape[0]
nb_actions = env.action_space.n
state = env.reset()
env.render()

done=False

for i in range(15):
    state = env.reset()
    score=0
    for j in range(500):
        env.render()
        action = np.random.randint(nb_actions)
        next_state, reward, done, _ = env.step(action)
        score += reward
        state = next_state
        time.sleep(0.001)
        if done:
        	break
    
    print("Score: {}".format(score))
env.close()