import gym
import torch

import utils
from agent import Agent

env = gym.make('LunarLander-v2')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device: ', device)

agent = Agent(state_size=env.observation_space.shape[0], action_size=env.action_space.n, device=device)

agent.qnet_local.load_state_dict(torch.load('checkpoint.pth'))

frames = agent.evaluate(env)
env.close()

utils.save_frames_as_gif(frames)
