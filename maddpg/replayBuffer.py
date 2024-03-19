import numpy as np

from pettingzoo.mpe import simple_adversary_v3
from agent import Agent

from typing import Dict, List

class replayBuffer:
    def __init__(self,mem_size: int, batch_size: int, dim_obs: Dict, dim_action: Dict):
        """
        Assuming dim_obs, dim_action both hold obs and action dimentions for each agent

        """
        self.counter = 0
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.num_agents = len(list(dim_action.keys()))
        self.dim_action = dim_action
        self.dim_obs = dim_obs

        self.dim_state = sum(list(dim_obs.values()))
        self.dim_action_all = sum(list(dim_action.values()))  
        
        self.agents = list(dim_obs.keys())        
        self.state_mem = np.zeros((mem_size, self.dim_state ))
        self.state_next_mem = np.zeros((mem_size, self.dim_state ))
        self.action_mem = np.zeros((mem_size, self.dim_action_all))

        self.termina_mem = np.zeros((mem_size, self.num_agents))
        self.trunc_mem = np.zeros((mem_size, self.num_agents))
        self.rewards_mem = np.zeros((mem_size, self.num_agents))
        self.init_actor_memmory()

    def init_actor_memmory(self):
        self.actor_state_memory = {}
        self.actor_new_state_memory = {}
        self.actor_action_memory = {}

        for agent in self.agents:
            self.actor_state_memory[agent] = np.zeros((self.mem_size, self.dim_obs[agent]))
            self.actor_new_state_memory[agent] = np.zeros((self.mem_size, self.dim_obs[agent]))
            self.actor_action_memory[agent] = np.zeros((self.mem_size, self.dim_action[agent]))        
        
    def store_buffer(self, obs: Dict, actions: Dict, rewards: Dict, obs_new: Dict, terminate: Dict, trunc: Dict):
        idx = self.counter % self.mem_size

        self.state_mem[idx] = np.concatenate(list(obs.values()), axis=0)
        self.state_next_mem[idx] = np.concatenate(list(obs_new.values()), axis=0)
        self.action_mem[idx] = np.concatenate(list(actions.values()), axis=0)

        self.termina_mem[idx] = list(terminate.values())
        self.trunc_mem[idx] = list(trunc.values())
        self.rewards_mem[idx] = list(rewards.values())

        for agent in self.agents:
            self.actor_state_memory[agent][idx] = obs[agent]
            self.actor_new_state_memory[agent][idx] = obs_new[agent]
            self.actor_action_memory[agent][idx] =  None # not need, since only need concatnated actions

        self.counter+=1

    def sample_batch(self):
        maxm_size = min(self.counter, self.mem_size)
        batch = np.random.choice(maxm_size, self.batch_size)

        state_batch = self.state_mem[batch]
        state_next_batch =  self.state_next_mem[batch]
        action_batch = self.action_mem[batch]
        reward_batch =  self.rewards_mem[batch]

        teminal_batch = self.termina_mem[batch]
        trunc_batch = self.trunc_mem[batch]

        actor_state_batch = {}
        actor_state_new_batch = {}
        actor_action_batch = {}
        for agent in self.agents:
            actor_state_batch[agent] = self.actor_state_memory[agent][batch]
            actor_state_new_batch[agent] = self.actor_new_state_memory[agent][batch]
            actor_action_batch[agent] = self.actor_action_memory[agent][batch]

        return state_batch, action_batch, reward_batch, state_next_batch, actor_state_batch, actor_state_new_batch, teminal_batch , trunc_batch




