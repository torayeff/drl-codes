import random
from time import time

import gym
import numpy as np
import torch

from agent import Agent

# seed for reproducibility
seed = 41
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

env = gym.make('LunarLander-v2')
env.seed(seed)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device: ', device)

agent = Agent(state_size=env.observation_space.shape[0],
              action_size=env.action_space.n,
              device=device,
              buffer_size=int(1e4),
              batch_size=64,
              gamma=0.99,
              tau=1e-3,
              lr=1e-3,
              update_every=4)

start = time()
scores = agent.train(env)
env.close()
print(f'Training time: {time() - start:.2f} seconds.')
