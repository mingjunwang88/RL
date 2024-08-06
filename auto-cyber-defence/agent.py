import torch as t
import torch.nn as nn
import torch.optim as optim
import numpy as np
from networks import Actor, Critic
from gymnasium.spaces import MultiDiscrete, Discrete, Box
from torch.distributions import Categorical
import torch.nn.functional as F
import copy

from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import BlueFlatWrapper, EnterpriseMAE
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent

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

class Agent:
    def __init__(self, dim_state, dim_action_all, dim_obs, fc1, fc2, fc3, num_action, lr, chkpt_dir=None, agent_idx=None):
        """
        dim_state: concatenation of each agent's obs
        dim_action_all: concatenation of each agent's action 
        dim_obs: local observation, each agent can have different of dim_obs
        num_action: number of actions, each agent can have different number of actions
        """
        self.device = t.device('cpu' if t.cuda.is_available() else 'cpu')
        self.agent_idx = agent_idx

        self.num_action = num_action
        name = f'agent_{agent_idx}'
        #                    dim_state, dim_action_all, fc1, fc2, lr):
        self.critic = Critic(dim_state, dim_action_all, fc1, fc2, fc3, lr, chkpt_dir, name+'_actor')
        self.actor =  Actor(dim_obs, num_action, fc1, fc2, fc3, lr, chkpt_dir, name+'_critic')
        #                   dim_obs, num_actions, fc1, fc2, lr):

    def choose_action(self, obs: np.array,):
        """
        obs: A numpy array from local observation
        In the future, it can be 2d array
        
        """
        obs = t.tensor(obs, dtype=t.float32)
        prob = F.softmax(self.actor(obs), dim=0)
        m = Categorical(prob)

        action = m.sample().to(self.device)
        log_prob = m.log_prob(action)

        return action.item(), log_prob

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
        prob = F.softmax(self.actor(obs), dim=0)
        m = Categorical(prob)

        action = m.sample().to(self.device)

        return action.item()

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_model(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()


