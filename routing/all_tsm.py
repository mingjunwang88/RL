#####################################
# Build a tabular version of qlearning
#####################################

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from itertools import product

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
        self.Qs[s,a] = self.Qs[s,a] + self.lr*(r + self.gamma*np.max(self.Qs[s_next,a]) - self.Qs[s,a])
        #self.Qs[s,a] = self.Qs[s,a] + self.lr*(r + self.gamma*self.Qs[s_next,a] - self.Qs[s,a])
        
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
        #Each episode need to go through all the nodes in the network
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
    
    def run_src_dest(self, src, dest):
        #run the path through src to dest
        self.q_agent.reset_memmory()
        total_reward = 0
        
        i=0
        s = src
        if src == dest:
            return total_reward
        while s!=dest:
            i+=1
            self.q_agent.remember_state(s)
            a = self.q_agent.act(s)
            s_next, r, done = self.env.step(a)
            
            r = -1. * r
            self.q_agent.update(s,a,r,s_next)
            
            total_reward+=r
            s=s_next
        return total_reward/i

    def plot(self, lst):
        plt.figure(figsize=(15,3))
        plt.title('Rewards with the episodes')
        plt.plot(lst)
        plt.show()
    
    def train_all_nodes(self, episodes):
        
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
    
    def train_src_dest(self,episodes):
        
        rewards= []
        lst = [i for i in range(self.env.n_stops)]
        pairs = [i for i in product(lst,lst)]
        lenth = len(pairs)
        
        for i in tqdm_notebook(range(episodes)):
            pair = pairs[np.random.choice(lenth)]
            reward = self.run_src_dest(pair[0],pair[1])
            rewards.append(reward)
            
        self.plot(rewards)
        return 
        
    def roll_out(self, state, optim=True):
        #state: first stop
        i = 0
        hist = []
        cost = 0.
        lst = [i for i in range(self.env.n_stops)]
        while i < self.env.n_stops:
            hist.append(state)
            
            Qs = np.copy(self.q_agent.Qs[state,:])
            Qs[hist] = -np.inf
            
            if optim:
                a = np.argmax(Qs)
            else:
                #a = np.random.choice(np.delete(Qs, hist)) #random select node not necessary the best one
                if len(hist) == self.env.n_stops:
                    a = state
                else:
                    a = np.random.choice(list(set(lst) - set(hist)))
            
            cost+=self.env.cost[state,a]
            s_next, r, done = self.env.step(a)
            
            state = s_next
            i+=1
        
        #output the best route from the frist stop
        return hist, cost
    
    def find_path(self,src, dest, optim=True):
        
        hist = []
        cost = 0.
        s = src
        if src == dest:
            print('Src and dest are same')
            return 
        while s != dest:
            hist.append(s)
            
            Qs = np.copy(self.q_agent.Qs[s,:])
            Qs[hist] = -np.inf
            
            if optim:
                a = np.argmax(Qs)
            else:
                a = np.random.choice(np.delete(Qs, hist)) #random select node not necessary the best one
            cost+=self.env.cost[s,a]
            s_next, r, done = self.env.step(a)
            
            s = s_next
        hist.append(dest)
        return hist,cost
                                            
                                    
            
            
            

            
            
            
            
            
        
    