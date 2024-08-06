import torch as t
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List 
import matplotlib.pyplot as plt
from typing import Dict, List
from gymnasium.spaces import MultiDiscrete, Discrete, Box
import copy
import pickle

from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import BlueFlatWrapper, EnterpriseMAE
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent

from agents.agent_mappo_cage4_indep import Agent
import os

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

#t.manual_seed(100)

class Mappo:
    def __init__(self, env, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init, steps_update,
                 tag, chkpt_dir=os.path.dirname(__file__)):
        """
        dim_state: concatenation of each agent's obs
        """
        self.env = env

        self.agents = {}   
        self.num_actions = {}
        self.lst = {}     

        for agent_index, agent in enumerate(env.agents):
            print('agent: ', agent)
            self.num_actions[agent] = env.action_space(agent).n
            #                          state_dim, obs_dim,                               action_dim,              lr_actor, lr_critic, gamma, K_epochs, eps_clip, 
            #                          has_continuous_action_space, chkpt_dir, name, action_std_init=0.6,)
            self.agents[agent] = Agent(sum([i.n for i in env.observation_space(agent)]), self.num_actions[agent], lr_actor, lr_critic, gamma, K_epochs, eps_clip, 
                                       has_continuous_action_space, action_std_init, chkpt_dir, 'ppo_indep' + str(int(agent_index)))
            self.lst[agent] = [np.zeros(i.n) for i in env.observation_space(agent)]

        self.gamma = gamma
        self.max_steps = 50
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

    def convert_state_to_numeric(self, obs: np.array, agent: str):
        """
        obs: A discrete verion of an state
        """
        lst_ = copy.deepcopy(self.lst[agent])
        for i,j in enumerate(obs):
            lst_[i][j] = 1
        final = np.concatenate(lst_)
        return final
    
    def create_mask(self, inp: str):
        return 'Invalid'.lower() not in inp.lower()

    def select_actions(self, obses: Dict) -> Dict:
        """
        Choose action for eqach agent
        obs: dict for each agent, assume the obs is discrete
        
        Note: if the observation is Discrete, it needs to be converted to numerical value
        """
        actions = {}
        for agent in self.env.agents:
            all_actions = env.action_labels(agent)
            masks = [self.create_mask(i) for i in all_actions] #not used 
            obs_numeric = obses[agent]
            actions[agent] = self.agents[agent].select_action(obs_numeric)
        return actions

    def plot_learning_curve(self,x, scores, figure_file='output/mappo_cage4_indep'):
        running_avg = np.zeros(len(scores))
        for i in range(len(running_avg)):
            running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
        plt.plot(x, running_avg)
        plt.title('Running average of previous 100 scores')

        tag = self.tag
        location = figure_file + f'_{tag}.png'
        #plt.savefig(location)
        plt.show()

    def learn(self, maxm_Iters, evaluate=False):
            reward_list = []
            steps = 0 

            if evaluate:
                self.load_check_point()                
            
            for i in range(maxm_Iters):
                steps_episodes = 0
                total_reward = 0
                obs, _ = self.env.reset()
                done_list = [False for j in self.env.agents]

                while not any(done_list):
                    steps_episodes+=1
                    steps+=1
                    if evaluate:
                        self.env.render()
                          
                    obs_ = {}
                    for agent in self.env.agents:
                        obs_[agent] = self.convert_state_to_numeric(obs[agent], agent)

                    actions = self.select_actions(obs_,)

                    obs_next, reward, terminate, trunc, _ = env.step(actions)
                    total_reward +=sum(list(reward.values())) / self.num_agents

                    all_states_next = np.concatenate(list(obs_next.values()), axis=0)
  
                    if steps_episodes >= self.max_steps:
                        for agent in self.env.agents:
                            terminate[agent], trunc[agent] = True, True

                    done={}
                    for agent in self.env.agents:
                        done[agent]=np.logical_or(terminate[agent], trunc[agent])
                        self.agents[agent].buffer.rewards.append(reward[agent])
                        self.agents[agent].buffer.is_terminals.append(done[agent])
                        
                        self.agents[agent].buffer.states_next.append(t.tensor(obs_next[agent], dtype=t.float))
                        self.agents[agent].buffer.all_states_next.append(t.tensor(all_states_next, dtype=t.float))

                    done_list = [i for i in list(done.values())]                        
                    obs = obs_next

                    # update PPO agent
                    if steps % self.steps_update == 0 and not evaluate:
                        #print('update!!')
                        for agent in self.env.agents:
                            self.agents[agent].update(tdzero=False)

                if i % 50 == 0:
                    print('iters:',i, 'total_reward: ', total_reward)
                #print('Eoisode: ', i, 'length: ', steps_episodes)

                #writer.add_scalar("Mean Reward", total_reward, i)
                #reward_list.append(total_reward)
                
                if not evaluate:
                    avg_score = np.mean(reward_list[-1:])
                    if avg_score > self.best_score:
                        self.save_check_point()
                        self.best_score = avg_score
                
            x = [i+1 for i in range(maxm_Iters)]
            self.plot_learning_curve(x, reward_list)
            writer.close()
            with open('output/mappo/'+ f'mappo_{self.tag}', 'wb') as f:
                pickle.dump(reward_list, f)


            print('---------Training Done---------')
"""
from pettingzoo.mpe import simple_adversary_v3, simple_crypto_v3, simple_push_v3, simple_world_comm_v3
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

print('DIM_STATE: ', DIM_STATE)
print('DIM_ACTION: ', DIM_ACTION)
"""
def env_creator_CC4(env_config: dict):
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=500
    )
    cyborg = CybORG(scenario_generator=sg)
    env = BlueFlatWrapper(env=cyborg, pad_spaces = False)
    return env
env = env_creator_CC4({})
#env = BlueFlatWrapper(env=cyborg, pad_spaces = False)
#env = EnterpriseMAE(env=cyborg, pad_spaces = False)
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


#            env, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init, steps_update,
                 #tag, chkpt_dir=os.path.dirname(__file__)):
for tag in range(25):
    mappo = Mappo(env, 1e-4,     1e-4,      0.95,  40,       0.2,      False,                        0.6,             100, tag)
    mappo.learn(maxm_Iters=1000, evaluate=False)
    mappo.load_check_point()

#python3 -m CybORG.Evaluation.evaluation --max-eps 2 . /tmp/output
# tensorboard --logdir=runs
"""
with open('mylist', 'rb') as f: 
...     mylist = pickle.load(f) 
"""
