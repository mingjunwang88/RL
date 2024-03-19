import torch as t
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, dim_state, fc1, fc2, n_agents, dim_action, lr):
        super(Critic, self).__init__()
        """
        critic_input: the concanation of obs from each agent. 
        note: all the agent have the sane dim_state since this is centralized critic
        
        """
        self.fc1 = nn.Linear(dim_state + dim_action*n_agents, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        """
        action: concatnation of each agent's action
        """
        state = state.to(self.device)
        action = action.to(self.device)

        state = state.to(self.device)
        action = action.to(self.device)

        x = F.relu(self.fc1(t.cat([state, action], axis=1)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    

class Actor(nn.Module):
    def __init__(self, dim_actor_obs, dim_action, fc1, fc2, lr):
        super(Actor, self).__init__()
        """
        dim_actor_obs: local observation might be different since this a decentralized actor

        """
        self.fc1 = nn.Linear(dim_actor_obs, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, dim_action)

        self.optimizer= optim.Adam(self.parameters(), lr=lr)
        self.device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = state.to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #x = t.softmax(x)

        return x

