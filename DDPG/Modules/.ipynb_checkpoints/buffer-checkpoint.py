import numpy as np

class Replaybuffer():
    def __init__(self, maxm_size, dim_action, dim_state):
        self.maxm_size = maxm_size
        self.dim_action = dim_action
        self.dim_state = dim_state
        self.counter = 0
        self.batch_size = 64
        
        self.states = np.zeros((maxm_size, dim_state), dtype=float)
        self.actions = np.zeros((maxm_size, dim_action), dtype=float)
        self.rewards= np.zeros((maxm_size,1), dtype=float)
        self.state_nexts = np.zeros((maxm_size, dim_state), dtype=float)
        self.dones = np.zeros((maxm_size,1), dtype=float)
        
    def store_transactions(self, state, action, reward, state_next, done):
        i = self.counter % self.maxm_size
        self.states[i] = state
        self.actions[i] = action
        self.rewards[i] = reward
        self.state_nexts[i] = state_next
        self.dones[i] = done
        
        self.counter+=1

    def sample_batch(self):
        minim = min([self.counter, self.maxm_size])
        batch = np.random.choice(minim, size=self.batch_size)
        
        state_batch = self.states[batch]
        action_batch = self.actions[batch]
        reward_batch = self.rewards[batch]
        state_next_batch = self.state_nexts[batch]
        done_batch = 1 - self.dones[batch]
        
        return state_batch, action_batch, reward_batch,state_next_batch, done_batch
 
        
        
        