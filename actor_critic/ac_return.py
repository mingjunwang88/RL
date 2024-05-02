import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

seed = 0
torch.manual_seed(seed)

class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self, dim_obs, dim_action, fc1):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(dim_obs, fc1)

        # actor's layer
        self.action_head = nn.Linear(fc1, dim_action)

        # critic's layer
        self.value_head = nn.Linear(fc1, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

        self.optimizer = optim.Adam(self.parameters(), lr=3e-3)
        self.eps = np.finfo(np.float32).eps.item()   

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = F.relu(self.affine1(x))

        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=0)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values

class Agent:
    def __init__(self, env, dim_obs, dim_action, fc1, gamma):
        self.policy = Policy(dim_obs, dim_action, fc1)
        self.env = env
        self.gamma = gamma

    def select_action(self, state):
        state = torch.from_numpy(state).float()
        probs, state_value = self.policy(state)

        # create a categorical distribution over the list of probabilities of actions
        m = Categorical(probs)

        # and sample an action using the distribution
        action = m.sample()

        # save to action buffer
        self.policy.saved_actions.append((m.log_prob(action), state_value))

        # the action to take (left or right)
        return action.item()

    def finish_episode(self, ):
        """
        Training code. Calculates actor and critic loss and performs backprop.
        """
        R = 0
        saved_actions = self.policy.saved_actions
        policy_losses = [] # list to save actor (policy) loss
        value_losses = [] # list to save critic (value) loss
        returns = [] # list to save the true values

        # calculate the true value using rewards returned from the environment
        for r in self.policy.rewards[::-1]:
            # calculate the discounted value
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + self.policy.eps)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

        # reset gradients
        self.policy.optimizer.zero_grad()

        # sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

        # perform backprop
        loss.backward()
        self.policy.optimizer.step()

        # reset rewards and action buffer
        del self.policy.rewards[:]
        del self.policy.saved_actions[:]

    def plot_learning_curve(self, x, scores, figure_file='ac_real_return.png'):
        running_avg = np.zeros(len(scores))
        for i in range(len(running_avg)):
            running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
        plt.plot(x, running_avg)
        plt.title('Running average of previous 100 scores')
        plt.savefig(figure_file)

    def learn(self, total_episodes = 400, render=False):
        running_reward = 10

        reward_list = []
        for i_episode in range(total_episodes):

            # reset environment and episode reward
            state, _ = self.env.reset()
            ep_reward = 0

            # for each episode, only run 9999 steps so that we don't
            # infinite loop while learning
            done = False
            while not done:
            #for t in range(1, 10000):
                # select action from policy
                action = self.select_action(state)

                # take the action
                state, reward, terminal, trunc, _ = self.env.step(action)
                done = np.logical_or(terminal, trunc)
                if render:
                    self.env.render()

                self.policy.rewards.append(reward)
                ep_reward += reward

                if terminal:
                    pass
                    #break

            # update cumulative reward
            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

            # perform backprop
            self.finish_episode()

            # log results
            if i_episode % 50 == 0:
                print('Episode: ', i_episode, ' running_reward: ', running_reward, ' ep_reward: ', ep_reward)

            reward_list.append(ep_reward)
        x = [i+1 for i in range(total_episodes)]
        self.plot_learning_curve(x, reward_list)


if __name__ == '__main__':
    #env = gym.make("MountainCar-v0")
    #env = gym.make('Acrobot-v1')
    env = gym.make('CartPole-v1')
    env.reset(seed=seed)

    dim_obs = env.observation_space.shape[0]
    dim_action = env.action_space.n

    agent = Agent(env, dim_obs, dim_action, 128, 0.99)
    agent.learn(total_episodes = 2000)
