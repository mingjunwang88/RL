import torch as t
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pettingzoo.mpe import simple_adversary_v3
from typing import Dict, List 
import matplotlib.pyplot as plt

from agent import Agent
from replayBuffer import replayBuffer
from typing import Dict, List

class Maddpg:
    def __init__(self, env, fc1, fc2, n_agents, dim_obs: Dict, dim_action: Dict, mem_size=1000000, batch_size=64, gamma=0.99, lr=0.001):
        """
        dim_obs: dict obs dimension for each agent
        dim_action: dic action dimension for each agent
        
        """

        self.env = env
        dim_state = sum([env.observation_space(agent).shape[0] for agent in env.agents])

        self.agents = {}        
        for agent in env.agents:
            #                          dim_state, dim_actor, fc1, fc2, n_agents, dim_action):
            self.agents[agent] = Agent(dim_state, dim_obs[agent], fc1, fc2, n_agents, dim_action[agent], lr)

        self.gamma = gamma
        #                          mem_size, batch_size, dim_obs, dim_action)
        self.buffer = replayBuffer(mem_size, batch_size, dim_obs, dim_action)
        self.batch_size = batch_size
        self.max_steps = 25
        self.num_agents = len(env.agents)

    def chose_actions(self, obses: Dict) -> Dict:
        """
        Chose action for each agent: adversary_0, agent_0, agent_1
        obs: dict for each agent
        
        """
        actions = {}
        for agent in self.env.agents:
            actions[agent] = self.agents[agent].choose_action(obses[agent])

        return actions

    def plot_learning_curve(self,x, scores, figure_file='test'):
        running_avg = np.zeros(len(scores))
        for i in range(len(running_avg)):
            running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
        plt.plot(x, running_avg)
        plt.title('Running average of previous 100 scores')
        plt.savefig('ddpg_v2.png')

    def learn(self, num_games=10,):
        return_list = []
        n=0

        for i in range(num_games):
            print('game: ', i)
            iters= 0
            total_reward = 0
            terminate_list = [False for i in self.env.agents]
            trun_list = [False for i in self.env.agents]
            obs, _ = self.env.reset()
            
            while not (any(terminate_list) or any(trun_list)): 
                #env.render()
                n+=1
                #if true, keep looping, if done is True, then stop
                # if any is true, then stop
                iters+=1
                actions = self.chose_actions(obs)
                obs_next, rewards, terminate, trunc, info = self.env.step(actions)
                total_reward+=sum(rewards.values())

                if iters >= self.max_steps-1:
                    for agent in self.env.agents:
                        terminate[agent] = True
                        trunc[agent] = True

                terminate_list = list(terminate.values())
                trun_list = list(trunc.values())

                self.buffer.store_buffer(obs, actions, rewards, obs_next, terminate, trunc)

                if self.buffer.counter >= self.batch_size:
                    #sample batch
                    state_batch, action_batch, reward_batch, state_next_batch, actor_state_batch, actor_state_new_batch, teminal_batch, trunc_batch = self.buffer.sample_batch()
                    
                    state_batch = t.tensor(state_batch, dtype=t.float32)
                    action_batch = t.tensor(action_batch, dtype=t.float32)
                    reward_batch = t.tensor(reward_batch, dtype=t.float32)
                    state_next_batch = t.tensor(state_next_batch, dtype=t.float32)

                    dones_batch= np.logical_or(teminal_batch, trunc_batch)

                    ##obtain the concatnation of agent actions first
                    actions_current_pi = []
                    actions_next_target = []

                    for j, agent in enumerate(self.env.agents):
                        action_1 = self.agents[agent].choose_action(actor_state_batch[agent])
                        actions_current_pi.append(action_1)
                        
                        action_2 = self.agents[agent].choose_action(actor_state_new_batch[agent], use_targ=True)
                        actions_next_target.append(action_2)

                    action_batch_current_pi = t.tensor(np.concatenate(actions_current_pi, axis=1), dtype=t.float32)
                    action_next_batch_targ = t.tensor(np.concatenate(actions_next_target, axis=1), dtype=t.float32)
                    
                    for i, agent in enumerate(self.env.agents):
                        ###update the critic
                        ##TODO: need to check if terminal or trunc
                        with t.no_grad():
                            target = t.squeeze(self.gamma*self.agents[agent].critic_t(state_next_batch, action_next_batch_targ))
                            target[dones_batch[:,i]] = 0
                            target = reward_batch[:,i] + target

                        Q = t.squeeze(self.agents[agent].critic(state_batch, action_batch))
                        critic_loss = ((target - Q)**2).mean()
                        self.agents[agent].critic.optimizer.zero_grad()
                        self.agents[agent].actor.optimizer.zero_grad()
                        critic_loss.backward()
                        self.agents[agent].critic.optimizer.step()
                        self.agents[agent].actor.optimizer.step()
                                     
                        ###update the actor
                        loss_actor = -self.agents[agent].critic(state_batch, action_batch_current_pi)
                        loss_actor = loss_actor.mean()
                        self.agents[agent].critic.optimizer.zero_grad()
                        self.agents[agent].actor.optimizer.zero_grad()
                        loss_actor.backward()
                        self.agents[agent].critic.optimizer.step()
                        self.agents[agent].actor.optimizer.step()

                if n % 200 == 0:
                    for agent in self.agents:
                        self.agents[agent].update_parms()
                obs = obs_next
            print('Total reward: ', total_reward)
            return_list.append(total_reward)

        x = [i+1 for i in range(num_games)]
        self.plot_learning_curve(x, return_list)

env = simple_adversary_v3.parallel_env(render_mode=None, continuous_actions=True)
env.reset()
AGENTS = env.agents
NUM_AGENTS = len(env.agents)

DIM_OBS = {}
DIM_ACTIONS = {}
for agent in env.agents:
    DIM_OBS[agent] = env.observation_space(agent).shape[0]
    DIM_ACTIONS [agent] = env.action_space(agent).shape[0]

DIM_STATE = sum(list(DIM_OBS.values()))
DIM_ACTION = sum(list(DIM_ACTIONS.values()))

print('dim_state: ', DIM_STATE)
print('dim_action: ', DIM_ACTION)
print('DIM_ACTIONS: ', DIM_ACTIONS)
print('DIM_OBS: ', DIM_OBS)
print()

#               env, fc1, fc2, n_agents, dim_obs: Dict, dim_action: Dict, mem_size=1000000, batch_size=64, gamma=0.99):
maddpg = Maddpg(env, 100, 64, len(env.agents), DIM_OBS, DIM_ACTIONS, batch_size=1024, mem_size=1000000, gamma=0.95, lr=0.01)      
maddpg.learn(num_games=500)
