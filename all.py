#####################################
# Build a tabular version of qlearning
#####################################

import numpy as np
from delivery import DeliveryEnvironment
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook

class QAgent():
    def __init__(self,num_state, num_action,lr=0.08,gamma=0.95, epsilon = 1.0, epsilon_min = 0.01,epsilon_decay = 0.999,):
        self.num_state = num_state
        self.num_action = num_action
        self.lr = lr
        self.Qs = np.zeros((num_state, num_action)) #inital the Q values
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def update(self, s, a, r, s_next):
        #self.Qs[s,a] = self.Qs[s,a] + self.lr*(r + self.gamma*np.max(self.Qs[s_next,a]) - self.Qs[s,a])
        self.Qs[s,a] = self.Qs[s,a] + self.lr*(r + self.gamma*self.Qs[s_next,a] - self.Qs[s,a])
        
        if self.epsilon > self.epsilon_min:
            self.epsilon*=self.epsilon_decay
    
    def act(self, s):
        qs = np.copy(self.Qs[s,:])        
        qs[self.state_memmory] = -np.inf

        if np.random.rand() > self.epsilon:
            a = np.argmax(qs)
        else:
            a = np.random.choice([x for x in range(self.num_action) if x not in self.state_memmory])
            
        return a
    
    def reset_memmory(self):
        self.state_memmory=[]
        
    def remember_state(self, s):
        self.state_memmory.append(s)
        

class Agent():
    def __init__(self, q_agent, env):
        self.q_agent = q_agent
        self.env = env
        
    def run_episode(self):
        s = self.env.reset()
        self.q_agent.reset_memmory()
        total_reward = 0
        
        i= 0
        while i < self.env.n_stops:
            self.q_agent.remember_state(s)
            a = self.q_agent.act(s)
            s_next, r, done = self.env.step(a)
            
            r = -1. * r
            self.q_agent.update(s,a,r,s_next)
            
            total_reward+=r
            i+=1
            s = s_next
            
            if done:
                break
        
        return total_reward
    
    def plot(self, lst):
        plt.figure(figsize=(15,3))
        plt.title('Rewards with the episodes')
        plt.plot(lst)
        plt.show()
    
    def train(self, episodes):
        
        rewards = []
        n=0
        for i in tqdm_notebook(range(episodes)):
            n+=1
            reward = self.run_episode()
            if n % self.env.render_each == 0:
                pass
                #self.env.render()
            rewards.append(reward)   
        self.plot(rewards)
        return 
    
    def roll_out(self, state):
        #state: first stop
        i = 0
        hist = []
        while i < self.env.n_stops:
            i+=1
            hist.append(state)
            
            Qs = np.copy(self.q_agent.Qs[state,:])
            Qs[hist] = -np.inf
            
            a = np.argmax(Qs)
            s_next, r, done = self.env.step(a)
            
            state = s_next
        
        #output the best route from the frist stop
        return hist
            
            
            
        
    
