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

from agent_mappo import Agent
import os

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

#t.manual_seed(100)

class Mappo:
    def __init__(self, env, dim_state: int, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init, steps_update,
                 tag, chkpt_dir=os.path.dirname(__file__)):
        """
        dim_state: concatenation of each agent's obs
        """
        self.env = env

        self.agents = {}   
        self.num_actions = {}
        self.lst = {}     

        for agent_index, agent in enumerate(env.agents):
            self.num_actions[agent] = env.action_space(agent).n
            #                          state_dim, obs_dim,                               action_dim,              lr_actor, lr_critic, gamma, K_epochs, eps_clip, 
            #                          has_continuous_action_space, chkpt_dir, name, action_std_init=0.6,)
            self.agents[agent] = Agent(dim_state, env.observation_space(agent).shape[0], self.num_actions[agent], lr_actor, lr_critic, gamma, K_epochs, eps_clip, 
                                       has_continuous_action_space, action_std_init, chkpt_dir, 'ppo_' + str(int(agent_index)))
        self.gamma = gamma
        self.max_steps = 24
        self.num_agents = len(env.agents)
        self.tag = tag
        self.best_score = -1000000
        self.loss_fn = nn.MSELoss()
        self.steps_update = steps_update

    def save_check_point(self):
        print('...save check point...')
        for agent in self.env.agents:
            self.agents[agent].save_model()

    def load_check_point(self):
        print('...load check point...')
        for agent in self.env.agents:
            self.agents[agent].load_model()

    def select_actions(self, obses: Dict, all_state: np.array) -> Dict:
        """
        Choose action for each agent
        obs: dict for each agent, assume the obs is discrete
        all_state: for calculating critic(state) value
        
        Note: if the observation is Discrete, it needs to be converted to numerical value
        """
        actions = {}
        for agent in self.env.agents:
            #obs_numeric = self.convert_state_to_numeric(obses[agent], agent)
            actions[agent] = self.agents[agent].select_action(obses[agent], all_state)
        return actions

    def plot_learning_curve(self,x, scores, figure_file='output/mappo_cyborg_return'):
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
            steps = 0 

            if evaluate:
                self.load_check_point()                
            
            for i in range(maxm_Iters):
                total_reward = 0
                obs, _ = self.env.reset()
                done_list = [False for i in self.env.agents]
            
                while not any(done_list):
                    steps+=1
                    if evaluate:
                        #self.env.render()
                        pass
                    
                    """
                    obs_ = {}
                    for agent in self.agents:
                        obs_[agent] = self.convert_state_to_numeric(obs[agent], agent)
                    """
                    all_state = np.concatenate(list(obs.values()), axis=0)
                    
                    actions = self.select_actions(obs, all_state)

                    obs_next, reward, terminate, trunc, _ = env.step(actions)
                    total_reward +=sum(list(reward.values()))

                    all_states_next = np.concatenate(list(obs_next.values()), axis=0)
  
                    if steps >= self.max_steps:
                        for agent in self.env.agents:
                            terminate[agent], trunc[agent] = True, True

                    done={}
                    for agent in self.env.agents:
                        done[agent]=np.logical_or(terminate[agent], trunc[agent])
                        self.agents[agent].buffer.rewards.append(reward)
                        self.agents[agent].buffer.is_terminals.append((done or trunc))
                        self.agents[agent].buffer.states_next.append(t.tensor(obs_next[agent], dtype=t.float))
                        self.agents[agent].buffer.all_states.append(t.tensor(all_states_next, dtype=t.float))

                    done_list = [i for i in list(done.values())]                           
                    obs = obs_next
                
                ### Update the policy!!!

                # update PPO agent
                if steps % self.steps_update == 0 and not evaluate:
                    for agent in self.env.agents:
                        self.agents[agent].update(tdzero=False)

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

from pettingzoo.mpe import simple_adversary_v3, simple_crypto_v3
#env = simple_adversary_v3.parallel_env(render_mode='Human', continuous_actions=False)
env = simple_crypto_v3.parallel_env(render_mode="human", continuous_actions=False)
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

#            env, dim_state, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init, steps_update,
                 #tag, chkpt_dir=os.path.dirname(__file__)):
mappo = Mappo(env, DIM_STATE, 1e-2,     1e-2,      0.95,  80,       0.2,      False,                        0.6,             4000, 0)
mappo.learn(maxm_Iters=100, evaluate=True)
mappo.load_check_point()

#python3 -m CybORG.Evaluation.evaluation --max-eps 2 . /tmp/output
# tensorboard --logdir=runs
