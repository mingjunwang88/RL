import torch as t
import torch.nn as nn
from torch.nn import Sequential
import torch.optim as optim
import torch.nn.functional as F
from gymnasium.spaces import MultiDiscrete, Discrete, Box
import os

def weights_init(m):
    if isinstance(m, nn.Linear):
        #t.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu') #relu, leaky_relu
        #t.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        #t.nn.init.orthogonal_(m.weights)
        t.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

class Critic_old(nn.Module):
    def __init__(self, dim_state, dim_action_all, fc1, fc2, fc3, lr):
        super(Critic, self).__init__()
        """
        Note: This is the centralzed critic, each agent has the same input:

        dim_state: concatenation of each agent's obs
        dim_action_all: concatnation of each agent's action 
        """

        self.fc1 = nn.Linear(dim_state + dim_action_all, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, fc3)
        self.fc4 = nn.Linear(fc3, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        self.device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state, action):

        """
        state: concatnation of each agent's observation 
        action: concatnation of each agent's action

        """
        state = state.to(self.device)
        action = action.to(self.device)

        z = t.cat([state, action])
        z = F.relu(self.fc1(z))
        z = F.relu(self.fc2(z))
        z = F.relu(self.fc3(z))
        z = self.fc4(z)

        return z

class Critic(nn.Module):
    def __init__(self, dim_state, dim_action_all, fc1, fc2, fc3, lr, chkpt_dir=None, name=None, init=False):
        super(Critic, self).__init__()
        """
        Note: This is the centralzed critic, each agent has the same input:

        dim_state: concatenation of each agent's obs
        dim_action_all: concatnation of each agent's action 
        """

        self.model = nn.Sequential(
            nn.Linear(dim_state + dim_action_all, fc1),
            nn.ReLU(),
            nn.Linear(fc1, fc2),
            nn.ReLU(),
            nn.Linear(fc2, fc3),
            nn.ReLU(),
            nn.Linear(fc3, 1)           
        )
        if init: 
            self.model.apply(weights_init)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        self.device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.chkpt_file = os.path.join(chkpt_dir, name)

    def save_checkpoint(self):
        t.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(t.load(self.chkpt_file))

    def forward(self, state, action):

        """
        state: concatnation of each agent's observation 
        action: concatnation of each agent's action

        """
        state = state.to(self.device)
        action = action.to(self.device)

        z = t.cat([state, action])
        z = self.model(z)

        return z

class Critic_stateOld(nn.Module):
    def __init__(self, dim_state, fc1, fc2, fc3, lr):
        super(Critic_state, self).__init__()
        """
        Note: This is the centralzed critic, each agent has the same input:

        dim_state: concatenation of each agent's obs
        dim_action_all: concatnation of each agent's action 
        """

        self.fc1 = nn.Linear(dim_state, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, fc3)
        self.fc4 = nn.Linear(fc3, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        self.device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):

        """
        state: concatnation of each agent's observation 
        action: concatnation of each agent's action

        """
        state = state.to(self.device)

        z = F.relu(self.fc1(state))
        z = F.relu(self.fc2(z))
        z = F.relu(self.fc3(z))
        z = self.fc4(z)

        return z

class Critic_state(nn.Module):
    def __init__(self, dim_state, fc1, fc2, fc3, lr):
        super(Critic_state, self).__init__()
        """
        Note: This is the centralzed critic, each agent has the same input:

        dim_state: concatenation of each agent's obs
        dim_action_all: concatnation of each agent's action 
        """

        self.model = nn.Sequential(
            nn.Linear(dim_state, fc1),
            nn.ReLU(),
            nn.Linear(fc1, fc2),
            nn.ReLU(),
            nn.Linear(fc2, fc3),
            nn.ReLU(),
            nn.Linear(fc3, 1)           
        )

        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        self.device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):

        """
        state: concatnation of each agent's observation 
        action: concatnation of each agent's action

        """
        state = state.to(self.device)

        z = self.model(state)

        return z

class Actor_old(nn.Module):
    def __init__(self, dim_obs, num_actions, fc1, fc2, fc3, lr):
        super(Actor, self).__init__()

        """
        dim_obs: local agent observation
        num_action: number of action categories
        
        """
        self.fc1 = nn.Linear(dim_obs, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.fc3 = nn.Linear(fc2, fc3)
        self.fc4 = nn.Linear(fc3, num_actions)

        self.optimizer= optim.Adam(self.parameters(), lr=lr)
        self.device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, obs):
        """
        State: Agent's local observation

        """
        x = obs.to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        #x = t.softmax(x)

        return x

class Actor(nn.Module):
    def __init__(self, dim_obs, num_actions, fc1, fc2, fc3, lr, chkpt_dir=None, name=None, init=False):
        super(Actor, self).__init__()

        """
        dim_obs: local agent observation
        num_action: number of action categories
        
        """             
        self.model = nn.Sequential(
            nn.Linear(dim_obs, fc1),
            nn.ReLU(),
            nn.Linear(fc1, fc2),
            nn.ReLU(),
            nn.Linear(fc2, fc3),
            nn.ReLU(),
            nn.Linear(fc3, num_actions)           
        )
        if init:
            self.model.apply(weights_init)

        self.optimizer= optim.Adam(self.parameters(), lr=lr)
        self.device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.chkpt_file = os.path.join(chkpt_dir, name)

    def save_checkpoint(self):
        t.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(t.load(self.chkpt_file))

    def forward(self, obs):
        """
        obs: Agent's local observation

        """ 

        x = self.model(obs)

        return x
