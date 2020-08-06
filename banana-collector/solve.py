import random
from time import time

import numpy as np
import torch
from unityagents import UnityEnvironment

from agent import Agent

# seed for reproducibility
seed = 41
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device: ', device)

# set up enviroment
env = UnityEnvironment(file_name="Banana.app")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

agent = Agent(state_size=brain.vector_observation_space_size,
              action_size=brain.vector_action_space_size,
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
