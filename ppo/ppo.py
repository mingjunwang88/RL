import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

import gymnasium as gym
#import roboschool
import torch as t 

from components import PPO
#PPO: state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6):
#env_name ='Pendulum-v1'
#env_name = 'MountainCarContinuous-v0'

env_name = 'CartPole-v1'
env = gym.make(env_name)

has_continuous_action_space = False

# state space dimension
state_dim = env.observation_space.shape[0]
# action space dimension
if has_continuous_action_space:
    action_dim = env.action_space.shape[0]
else:
    action_dim = env.action_space.n

lr_actor = 0.0003 #0.0003       # learning rate for actor network
lr_critic = 0.001       # learning rate for critic network
eps_clip = 0.2          # clip parameter for PPO
gamma = 0.99            # discount factor
K_epochs = 80 #80            # update policy for K epochs in one PPO update
steps_update = 4000
num_episodes = 2000
steps_print = 100

def train():
    ppo = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6)
    
    time_step = 0
    num_rewards = []
     # training loop
    for i in range(num_episodes):
        
        done = False
        trun = False 
        state, _ = env.reset()
        current_ep_reward = 0

        while not (done or trun):

            # select action with policy
            action = ppo.select_action(state)
            state_next, reward, done, trun, _ = env.step(action)

            # saving reward and is_terminals
            ppo.buffer.rewards.append(reward)
            ppo.buffer.is_terminals.append(done)
            ppo.buffer.states_next.append(t.tensor(state_next, dtype=t.float))

            time_step +=1
            current_ep_reward += reward

            # update PPO agent
            if time_step % steps_update == 0:
                ppo.update(tdzero=False)

            state = state_next

            """
            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)
            """
        num_rewards.append(current_ep_reward)
        if i % steps_print == 0:
            print(f'episode: {i}: ', current_ep_reward)

    x = [i+1 for i in range(num_episodes)]
    ppo.plot_learning_curve(x, num_rewards)


if __name__ == '__main__':

    train()
        

        
