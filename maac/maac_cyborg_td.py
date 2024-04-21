import torch as t
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List 
import matplotlib.pyplot as plt
from agent import Agent
from typing import Dict, List
from gymnasium.spaces import MultiDiscrete, Discrete, Box
import copy

from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import BlueFlatWrapper, EnterpriseMAE
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

#t.manual_seed(100)

class Maac:
    def __init__(self, env, dim_state: int, dim_action_all:int, fc1, fc2, fc3,  gamma, lr, tag,  chkpt_dir='output/models/'):
        """
        dim_state: concatenation of each agent's obs
        dim_action_all: concatenation of each agent's action 
        """
        self.env = env
        ### Each observation_space is a MultiDiscrete
        #dim_state = sum([sum(list(env.observation_space(agent))) for agent in env.agents])
        ### Each action_space is Discrete
        #dim_action_all = sum([env.action_spaces()[agent].n for agent in env.agents])

        self.agents = {}   
        self.num_actions = {}
        self.lst = {}     
        for agent_index, agent in enumerate(env.agents):
            #                          dim_state, dim_action_all, dim_obs, fc1, fc2, fc3, num_action, lr, chkpt_dir=None, agent_idx=None):
            self.agents[agent] = Agent(dim_state, dim_action_all, sum([i.n for i in env.observation_space(agent)]), 
                                       fc1, fc2, fc3, env.action_space(agent).n, lr, chkpt_dir, agent_index)
            self.num_actions[agent] =env.action_space(agent).n
            self.lst[agent] = [np.zeros(i.n) for i in env.observation_space(agent)]

        self.gamma = gamma
        self.max_steps = 100
        self.num_agents = len(env.agents)
        #self.num_actions = env.action_space('blue_agent_0').n
        self.tag = tag
        self.best_score = -100000
        self.loss_fn = nn.MSELoss()

    def save_check_point(self):
        print('...save check point...')
        for agent in self.env.agents:
            self.agents[agent].save_models()

    def load_check_point(self):
        print('...load check point...')
        for agent in self.env.agents:
            self.agents[agent].load_model()
        
    def convert_state_to_numeric(self, obs: np.array, agent: str):
        """
        obs: A discrete verion of an state
        """
        lst_ = copy.deepcopy(self.lst[agent])
        for i,j in enumerate(obs):
            lst_[i][j] = 1
        final = np.concatenate(lst_)
        return final

    def conver_action_to_numeric(self, action, agent: str):
        num_actions = self.num_actions[agent]
        action_ = np.zeros(num_actions)
        action_[action] = 1.0

        return action_


    def choose_actions(self, obses: Dict) -> Dict:
        """
        Choose action for each agent
        obs: dict for each agent, assume the obs is discrete
        Note: if the observation is Discrete, it needs to be converted to numerical value
        """
        actions = {}
        log_probs = {}

        for agent in self.env.agents:
            obs_numeric = self.convert_state_to_numeric(obses[agent], agent)
            actions[agent], log_probs[agent] = self.agents[agent].choose_action(obs_numeric)
        return actions, log_probs

    def plot_learning_curve(self,x, scores, figure_file='output/maac_cyborg'):
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
            
            for i in range(maxm_Iters):
                steps = 0 
                total_reward = 0
                obs, _ = self.env.reset()
                done_list = [False for i in self.env.agents]
            
                while not any(done_list):
                    steps+=1
                    actions, log_probs = self.choose_actions(obs)

                    obs_next, reward, terminate, trunc, _ = env.step(actions)
                    total_reward +=sum(list(reward.values()))

                    ### Need to choose the next actions just for computing the target value
                    actions_next, log_probs_next = self.choose_actions(obs_next)
  
                    if steps >= self.max_steps:
                        for agent in self.env.agents:
                            terminate[agent], trunc[agent] = True, True

                    done={}
                    for agent in self.env.agents:
                        done[agent]=np.logical_or(terminate[agent], trunc[agent])
                    
                    done_list = [i for i in list(done.values())]                    

                    ##convert discrete obs in to coninouse space!
                    obs_ = {}
                    obs_next_ = {}
                    for agent in self.agents:
                        obs_[agent] = self.convert_state_to_numeric(obs[agent], agent)
                        obs_next_[agent] = self.convert_state_to_numeric(obs_next[agent], agent)

                    ## Update each agent
                    state_all = np.concatenate(list(obs_.values()))
                    state_all = t.tensor(state_all, dtype=t.float32)

                    state_all_next = np.concatenate(list(obs_next_.values()))
                    state_all_next = t.tensor(state_all_next, dtype=t.float32)

                    action_all = [self.conver_action_to_numeric(actions[agent], agent) for agent in self.env.agents]
                    action_all = t.tensor(np.concatenate(action_all), dtype=t.float32)

                    action_next_all = [self.conver_action_to_numeric(actions_next[agent], agent) for agent in self.env.agents]
                    action_next_all = t.tensor(np.concatenate(action_next_all), dtype=t.float32)

                    for agent in self.env.agents:   
                        
                        ## This is centralized critic
                        target = t.tensor(reward[agent]) + self.gamma * self.agents[agent].critic(state_all_next, action_next_all)*(1-int(done[agent]))
                        action_value = self.agents[agent].critic(state_all, action_all)
                        #diff = target - action_value
                
                        ## Loss for critic
                        loss_critic = self.loss_fn(target,action_value)    #diff**2
                        
                        ## Loss for actor
                        loss_actor = -log_probs[agent] * action_value

                        self.agents[agent].actor.zero_grad()
                        self.agents[agent].critic.zero_grad()

                        (loss_critic + loss_actor).backward()
                        
                        self.agents[agent].actor.optimizer.step()
                        self.agents[agent].critic.optimizer.step()
                    
                    obs = obs_next
                    #actions = actions_next
                reward_list.append(total_reward)
                
                if i % 50 == 0:
                    print('total_reward: ', total_reward)
                
                if not evaluate:
                    #avg_score = np.mean(reward_list[-10:])
                    avg_score = total_reward / steps
                    if avg_score > self.best_score:
                        self.save_check_point()
                        self.best_score = avg_score
                writer.add_scalar("Mean Reward", total_reward, i)

            x = [i+1 for i in range(maxm_Iters)]
            self.plot_learning_curve(x, reward_list)
            writer.close()
            print('---------Training Done---------')
"""
env = simple_adversary_v3.parallel_env(render_mode='Human', continuous_actions=False)
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
"""

#sg = EnterpriseScenarioGenerator()
#cyborg = CybORG(scenario_generator=sg)
#env = EnterpriseMAE(env=cyborg, pad_spaces = False)

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

AGENTS = env.agents
NUM_AGENTS = len(env.agents)

DIM_OBS = {}
DIM_ACTIONS = {}
for agent in env.agents:
    DIM_OBS[agent] = sum([i.n for i in env.observation_space(agent)])
    DIM_ACTIONS [agent] = env.action_space(agent).n

DIM_STATE = sum(list(DIM_OBS.values()))
DIM_ACTION = sum(list(DIM_ACTIONS.values()))
print('DIM_STATE: ', DIM_STATE)
print('DIM_ACTION: ', DIM_ACTION)

if True:
    maac = Maac(env, DIM_STATE, DIM_ACTION, 500, 250, 126, gamma=0.99, lr=0.01, tag=0)    
    maac.learn(maxm_Iters=200)
    maac.load_check_point()

#python3 -m CybORG.Evaluation.evaluation --max-eps 2 . /tmp/output
# tensorboard --logdir=runs
