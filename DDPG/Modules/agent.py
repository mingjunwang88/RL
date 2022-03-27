from buffer import Replaybuffer
from networks import Critic, Actor
import copy 
import gym
import torch as t
import numpy as np
import torch.nn as nn
import torch.nn.functional as f

class Agent():
    def __init__(self, dim_state, dim_action,maxm_size,env, maxm_iters ):
        
        self.critic = Critic(dim_state, dim_action)
        self.critic_t = copy.deepcopy(self.critic) 
        self.actor = Actor(dim_state, dim_action)
        self.actor_t = copy.deepcopy(self.actor)
        self.buffer = Replaybuffer(maxm_size, dim_action, dim_state)
        
        self.env = env 
        self.maxm_iters = maxm_iters
        
    def learn(self):
        rewards_list = []
        n_steps = 0
        
        for i in range(self.maxm_iters):
            total_reward = 0   
            state = self.env.reset()
            done = False
            while not done:
                n_steps+=1
                action = self.actor(t.tensor(state)).cpu().detach().numpy()
                #action = self.env.action_space.sample()
                action += np.random.normal(0, 0.1, size=env.action_space.shape[0])
                state_next, reward, done, _ = self.env.step(action)
                self.buffer.store_transactions(state, action, reward,state_next, done)
                                               #state, action, reward, state_next, done):
                total_reward+=reward
                
                #Update the model
                if self.buffer.counter >=self.buffer.batch_size:
                    state_batch, action_batch, rewards_batch, stats_new_batch, done_batch = self.buffer.sample_batch()

                    state_batch = t.tensor(state_batch, dtype=t.float)
                    action_batch = t.tensor(action_batch, dtype=t.float)
                    rewards_batch = t.tensor(rewards_batch, dtype=t.float)
                    stats_new_batch = t.tensor(stats_new_batch, dtype=t.float)
                    done_batch = t.tensor(done_batch, dtype=t.float)
                    
                    #Critic update
                    targets = rewards_batch + 0.99 * self.critic_t(stats_new_batch, self.actor_t(stats_new_batch)) * done
                    preds = self.critic(state_batch, action_batch)
                    loss_critic = f.mse_loss(targets, preds)
                    self.critic.optimizer.zero_grad()
                    loss_critic.backward()
                    self.critic.optimizer.step()
                    
                    #Actor update
                    loss_actor = -self.critic(state_batch, self.actor(state_batch)).mean()
                    self.actor.optimizer.zero_grad()
                    loss_actor.backward()
                    self.actor.optimizer.step()
                    if n_steps % 1000 ==0:
                        self.critic_t = copy.deepcopy(self.critic)
                        self.actor_t = copy.deepcopy(self.actor)
                    state = state_next
            print(total_reward)                 
                
env = gym.make('Pendulum-v1')
#env = gym.make('MountainCarContinuous-v0')
dim_state = env.observation_space.shape[0]
dim_action = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
print('dim_state: ', dim_state)
print ('dim_action: ', dim_action)
print('action sample: ', env.action_space.sample())
agent= Agent(dim_state,dim_action,100000, env, 2000)
agent.learn()
        