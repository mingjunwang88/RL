import torch as t
import torch.nn as nn
import torch.optim as optim
import numpy as np
from networks import Actor, Critic

class Agent:
    def __init__(self, dim_state: int, dim_actor: int, fc1, fc2, n_agents, dim_action, lr):
        self.dim_state = dim_state
        self.n_agents = n_agents
        self.dim_action = dim_action
        #                    dim_state, fc1, fc2, n_agents, dim_action):
        self.critic = Critic(dim_state, fc1, fc2, n_agents, dim_action, lr)
        self.actor = Actor(dim_actor, dim_action, fc1, fc2, lr)

        self.critic_t = Critic(dim_state, fc1, fc2, n_agents, dim_action, lr)
        self.actor_t = Actor(dim_actor, dim_action, fc1, fc2, lr)
        #                    dim_actor_obs, dim_action, fc1, fc2):
        self.tau = 0.01

    def choose_action(self, obs: np.array, use_targ=False):
        """
        obs: A numpy array from local observation
        
        """
        obs = t.tensor(obs, dtype=t.float32)

        if use_targ:
            actions = self.actor_t(obs)
        else:
            actions = self.actor(obs)

        noise = t.rand(self.dim_action)
        actions = t.clamp((actions + noise), min=0, max=1)

        return actions.detach().cpu().numpy()
    
    def update_parms(self):
        
        critic_dict = dict(self.critic.named_parameters())
        critic_t_dict = dict(self.critic_t.named_parameters())
        
        actor_t_dict = dict(self.actor_t.named_parameters())
        actor_dict = dict(self.actor.named_parameters())
        
        #Update the critic
        for i in critic_dict:
            critic_dict[i] = self.tau*critic_dict[i].clone() + (1-self.tau)*critic_t_dict[i].clone()
        self.critic_t.load_state_dict(critic_dict)
        
        #Update the actor
        for j in actor_dict:
            actor_dict[j] = self.tau*actor_dict[j].clone() + (1-self.tau)*actor_t_dict[j].clone()
        self.actor_t.load_state_dict(actor_dict)


