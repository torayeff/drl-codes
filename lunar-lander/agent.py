import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Model
from replaybuffer import ReplayBuffer


class Agent:
    def __init__(self, state_size, action_size, device,
                 buffer_size=int(1e5), batch_size=64, gamma=0.99,
                 tau=1e-3, lr=5e-4, update_every=4):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.update_every = update_every

        # model settings
        self.qnet_local = Model(state_size, action_size).to(self.device)
        self.qnet_target = Model(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.qnet_local.parameters(), lr=self.lr)

        # replay buffer settings
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.update_step = 0

    def step(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

        self.update_step = (self.update_step + 1) % self.update_every
        if (self.update_step == 0) and (len(self.replay_buffer) > self.batch_size):
            experiences = self.replay_buffer.sample()
            self.learn(experiences)

    def act(self, state, eps=0.0):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        self.qnet_local.eval()
        with torch.no_grad():
            action_values = self.qnet_local(state)
        self.qnet_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return np.random.choice(self.action_size)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # convert to tensors and send to device
        states = torch.from_numpy(states).float().to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)

        # max returns max values (0) and indices (1)
        # unsqueeze is needed to add batch dim B x 1
        q_max = self.qnet_target(next_states).detach().max(1)[0].unsqueeze(1)
        y = rewards + self.gamma * q_max * (1 - dones)

        # select action values corresponding to actions
        # this is what .gather does
        # note for the expected we pass states, not next_states
        q_expected = self.qnet_local(states).gather(1, actions)

        loss = F.mse_loss(q_expected, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update()

    def soft_update(self):
        for target_param, local_param in zip(self.qnet_target.parameters(), self.qnet_local.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1 - self.tau) * target_param.data)

    def train(self, env, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
        scores = []
        scores_window = deque(maxlen=100)
        eps = eps_start

        for i_episode in range(1, n_episodes + 1):
            state = env.reset()
            score = 0
            for t in range(max_t):
                action = self.act(state, eps)
                next_state, reward, done, _ = env.step(action)
                self.step(state, action, reward, next_state, done)
                state = next_state
                score += reward

                if done:
                    break
            scores_window.append(score)
            scores.append(score)
            avg_scores = np.mean(scores_window)
            eps = max(eps_end, eps_decay * eps)

            print(f'\rEpisode {i_episode}\tAverage Score: {avg_scores:.2f}', end='')
            if i_episode % 100 == 0:
                print(f'\rEpisode {i_episode}\tAverage Score: {avg_scores:.2f}')
            if avg_scores >= 200.0:
                print(f'\nEnvironment solved in {i_episode - 100} episodes!'
                      f'\tAverage Score: {np.mean(scores_window):.2f}')
                torch.save(self.qnet_local.state_dict(), 'checkpoint.pth')
                break

        return scores

    def evaluate(self, env):
        frames = []
        state = env.reset()
        score = 0
        for i in range(2000):
            action = self.act(state)
            frames.append(env.render(mode='rgb_array'))
            state, reward, done, _ = env.step(action)
            score += reward
            if done:
                break
        print(f'Total score: {score:.2f}')

        return frames
