import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import os

from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import BlueFlatWrapper, EnterpriseMAE
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent

seed = 0
#torch.manual_seed(seed)
def env_creator_CC4(env_config: dict):
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=500
    )
    cyborg = CybORG(scenario_generator=sg)
    env = EnterpriseMAE(env=cyborg, pad_spaces = False)
    return env
env = env_creator_CC4({})
#env = BlueFlatWrapper(env=cyborg, pad_spaces = True)
obs, _ = env.reset()

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu') #relu, leaky_relu
        #t.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        #t.nn.init.orthogonal_(m.weights)
        #t.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

class Policy(nn.Module):
    """
    implements both actor and critic in one model, critic is state value
    """
    def __init__(self, dim_obs, num_action, fc1, lr, chkpt_dir=None, name=None):
        super(Policy, self).__init__()
        """
        dim_state: this is cenrtalized critic, so it concatinates all local obs
        
        """
        # actor's layer
        self.affine_p = nn.Linear(dim_obs, fc1)
        self.action_head = nn.Linear(fc1, num_action)

        # critic's layer
        self.affine_v = nn.Linear(dim_obs, fc1)
        self.value_head = nn.Linear(fc1, 1)

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        self.chkpt_file = os.path.join(chkpt_dir, name)
        print('self.chkpt_file: ', self.chkpt_file)

        self.optimizer = optim.Adam(self.parameters(), lr=lr) #original: lr=3e-3 --mj
        self.eps = np.finfo(np.float32).eps.item()   

    def forward(self, x):
        """
        forward for actor
        x: local obs
        """
        # actor: choses action to take from state s_t
        # by returning probability of each action
        x_p = F.relu(self.affine_p(x))
        action_prob = F.softmax(self.action_head(x_p), dim=0)

        # critic: evaluates being in the state s_t
        x_v = F.relu(self.affine_v(x))
        state_values = self.value_head(x_v)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values
    
    def value(self, x):
        """
        x: local obs
        """
        x = F.relu(self.affine_v(x))

        state_values = self.value_head(x)
        
        return state_values
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chkpt_file))

class Agent:
    def __init__(self, dim_obs: int, num_action: int, fc1, gamma, lr, chkpt_dir=None, agent_idx=None):
        name = f'agent_{agent_idx}'
        #                    dim_state, dim_obs, num_action, fc1, gamma, lr, chkpt_dir=None, name=None):
        self.policy = Policy(dim_obs, num_action, fc1, lr, chkpt_dir, name + '_policy')
        self.gamma = gamma
        self.agent_idx = agent_idx

    def select_action(self, state: np.array,):
        """
        state: Local observation

        States are transformed to numerical value!
        
        """
        state = torch.from_numpy(state).float()
        probs, state_value = self.policy(state)

        # create a categorical distribution over the list of probabilities of actions
        m = Categorical(probs)

        # and sample an action using the distribution
        action = m.sample()

        # save to action buffer
        self.policy.saved_actions.append((m.log_prob(action), state_value))

        # the action to take
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
            #value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))
            #value_losses.append(self.loss_fn(value, torch.tensor([R])))
            value_losses.append( torch.sqrt((value-torch.tensor([R]))**2) )

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

    def convert_state_to_numeric(self, obs: np.array,):
        """
        obs: A discrete verion of an state
        """
        lst_ = [np.zeros(i.n) for i in env.observation_space(env.agents[self.agent_idx])]
        for i,j in enumerate(obs):
            lst_[i][j] = 1
        final = np.concatenate(lst_)
        return final
    
    def get_action(self, observation, action_space):
        """ Gets the agent's action for that step. This is only for submmision!!!!!

        The function gets an action from the agent that should be performed based on the agent's internal state and provided observation and action space. 
        The contents is left empty to be overwritten by the class that inherits BaseAgent. 

        Parameters
        ----------
        observation : dict
            the 'data' dictionary contained within the Observation object
        action_space : dict
            a dictionary representation of the Action_Space object
        """
        obs = t.tensor(self.convert_state_to_numeric(observation), dtype=t.float32)
        prob = F.softmax(self.policy(obs)[0], dim=0)
        m = Categorical(prob)

        action = m.sample()

        return action.item()
    
    def save_models(self):
        self.policy.save_checkpoint()

    def load_model(self):
        self.policy.load_checkpoint()
