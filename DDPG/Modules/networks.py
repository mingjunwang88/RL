import torch as t
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f

class Critic(nn.Module):
    def __init__(self, dim_state, dim_action,):
        super(Critic, self).__init__()
        
        self.linear1 = nn.Linear(dim_state+dim_action, 400)
        self.linear2 = nn.Linear(400, 300)
        self.linear3 = nn.Linear(300, 1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=0.00001, weight_decay=1e-4)
        self.float()
        
    
    def forward(self, states, actions):
        output = f.relu(self.linear1(t.cat((states, actions), 1)))
        output = f.relu(self.linear2(output))
        output = t.tanh(self.linear3(output))
        
        return output
    
class Actor(nn.Module):
    def __init__(self, dim_state, dim_action):
        super(Actor, self).__init__()
        
        self.linear1 = nn.Linear(dim_state, 400)
        self.linear2 = nn.Linear(400, 300)
        self.linear3 = nn.Linear(300, dim_action)
        self.float()
        
        self.optimizer = optim.Adam(self.parameters(), lr=0.00001, weight_decay=1e-4)
        
    def forward(self, states):
        output = f.relu(self.linear1(states))
        output = f.relu(self.linear2(output))
        output = t.tanh(self.linear3(output))
        
        return output
