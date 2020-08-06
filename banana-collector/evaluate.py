import torch
from unityagents import UnityEnvironment

from agent import Agent

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Device: ', device)

# set up enviroment
env = UnityEnvironment(file_name="Banana.app")
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

agent = Agent(state_size=brain.vector_observation_space_size, action_size=brain.vector_action_space_size, device=device)

agent.qnet_local.load_state_dict(torch.load('checkpoint.pth'))

agent.evaluate(env)
env.close()
