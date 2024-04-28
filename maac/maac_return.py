import torch as t
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List 
import matplotlib.pyplot as plt
from typing import Dict, List
from gymnasium.spaces import MultiDiscrete, Discrete, Box
import copy

from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import BlueFlatWrapper, EnterpriseMAE
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent

from agent_return import Agent
import os

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

#t.manual_seed(100)

class Maac:
    def __init__(self, env, dim_state: int, fc1, fc2, gamma, lr, tag, chkpt_dir=os.path.dirname(__file__)):
        """
        dim_state: concatenation of each agent's obs
        dim_action_all: concatenation of each agent's action 
        """
        self.env = env

        self.agents = {}   
        self.num_actions = {}
        self.lst = {}     

        for agent_index, agent in enumerate(env.agents):
            self.num_actions[agent] = env.action_space(agent).n
            #                          dim_state, dim_obs,                                          num_action ,             fc1, fc2, gamma, lr, chkpt_dir=None, agent_idx=None):
            self.agents[agent] = Agent(dim_state, env.observation_space(agent).shape[0], self.num_actions[agent], fc1, fc2, gamma, lr, chkpt_dir, agent_index)
            #self.lst[agent] = [np.zeros(i.n) for i in env.observation_space(agent)]

        self.gamma = gamma
        self.max_steps = 24
        self.num_agents = len(env.agents)
        self.tag = tag
        self.best_score = -1000000
        self.loss_fn = nn.MSELoss()

    def save_check_point(self):
        print('...save check point...')
        for agent in self.env.agents:
            self.agents[agent].save_models()

    def load_check_point(self):
        print('...load check point...')
        for agent in self.env.agents:
            self.agents[agent].load_model()

    def create_mask(self, inp: str):
        return 'Invalid'.lower() not in inp.lower()

    def select_actions(self, obses: Dict, all_state: np.array) -> Dict:
        """
        Choose action for each agent
        obs: dict for each agent, assume the obs is discrete
        all_state: for caaulating critic(state) value
        
        Note: if the observation is Discrete, it needs to be converted to numerical value
        """
        actions = {}
        for agent in self.env.agents:
            #obs_numeric = self.convert_state_to_numeric(obses[agent], agent)
            actions[agent] = self.agents[agent].select_action(obses[agent], all_state, masks=None)
        return actions

    def plot_learning_curve(self,x, scores, figure_file='output/maac_return'):
        running_avg = np.zeros(len(scores))
        for i in range(len(running_avg)):
            running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
        plt.plot(x, running_avg)
        plt.title('Running average of previous 100 scores')

        tag = self.tag
        location = figure_file + f'_{tag}.png'
        plt.savefig(location)
        plt.show()

    def learn(self, maxm_Iters, evaluate=False):
            reward_list = []

            if evaluate:
                self.load_check_point()                
            
            for i in range(maxm_Iters):
                steps = 0 
                total_reward = 0
                obs, _ = self.env.reset()
                done_list = [False for i in self.env.agents]
            
                while not any(done_list):
                    steps+=1
                    if evaluate:
                        self.env.render()
                    
                    """
                    obs_ = {}
                    for agent in self.agents:
                        obs_[agent] = self.convert_state_to_numeric(obs[agent], agent)
                    """
                    all_state = np.concatenate(list(obs.values()), axis=0)
                    
                    actions = self.select_actions(obs, all_state)

                    obs_next, reward, terminate, trunc, _ = env.step(actions)
                    total_reward +=sum(list(reward.values()))
  
                    if steps >= self.max_steps:
                        for agent in self.env.agents:
                            terminate[agent], trunc[agent] = True, True

                    done={}
                    for agent in self.env.agents:
                        done[agent]=np.logical_or(terminate[agent], trunc[agent])
                        self.agents[agent].policy.rewards.append(reward[agent]) 
                    
                    done_list = [i for i in list(done.values())]                           
                    obs = obs_next
                
                ### Update the policy!!!
                if not evaluate:
                    for agent in self.env.agents:
                        self.agents[agent].finish_episode()
                
                writer.add_scalar("Mean Reward", total_reward, i)
                reward_list.append(total_reward)
                
                if i % 50 == 0:
                    print('iters:',i, 'total_reward: ', total_reward)

                if not evaluate:
                    avg_score = np.mean(reward_list[-1:])
                    if avg_score > self.best_score:
                        self.save_check_point()
                        self.best_score = avg_score
                
            x = [i+1 for i in range(maxm_Iters)]
            self.plot_learning_curve(x, reward_list)
            writer.close()
            print('---------Training Done---------')

from pettingzoo.mpe import simple_adversary_v3, simple_crypto_v3, simple_push_v3
#env = simple_adversary_v3.parallel_env(render_mode='Human', continuous_actions=False)
#env = simple_crypto_v3.parallel_env(render_mode="human", continuous_actions=False)
env = simple_push_v3.parallel_env(render_mode="human", continuous_actions = False)
env.reset()
AGENTS = env.agents
NUM_AGENTS = len(env.agents)

DIM_OBS = {}
DIM_ACTIONS = {}
for agent in env.agents:
    DIM_OBS[agent] = env.observation_space(agent).shape[0]
    DIM_ACTIONS [agent] = env.action_space(agent).n

DIM_STATE = sum(list(DIM_OBS.values()))
DIM_ACTION = sum(list(DIM_ACTIONS.values()))

print('DIM_STATE: ', DIM_STATE)
print('DIM_ACTION: ', DIM_ACTION)

#           env, dim_state, fc1, fc2, gamma,      lr, tag,  chkpt_dir='output/models/'):
maac = Maac(env, DIM_STATE, 126, 126, gamma=0.99, lr=2e-2, tag=0)    #3e-3
maac.learn(maxm_Iters=500, evaluate=False)
maac.load_check_point()

#python3 -m CybORG.Evaluation.evaluation --max-eps 2 . /tmp/output
# tensorboard --logdir=runs
