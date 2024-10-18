#####################################
# Build a tabular version of qlearning
#####################################

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from itertools import product
import gym

class QAgent():
    def __init__(self,env, num_state, num_action,lr=0.08,gamma=0.95, epsilon = 1.0, epsilon_min = 0.01,epsilon_decay = 0.999, maxm_iters=100):
        self.num_state = num_state
        self.num_action = num_action
        self.lr = lr
        self.Qs = np.zeros((num_state, num_action)) #inital the Q values
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.maxm_iters = maxm_iters
        self.env = env

    def update(self, s, a, r, s_next):
        actions = self.available_actions(s_next) 
        avail_actions = [i for i in actions if i not in self.state_memmory]
        
        if len(avail_actions)>0:
            a_next = np.random.choice([i for i in actions if i not in self.state_memmory])
        else:
            a_next = np.random.choice(actions)
        
        self.Qs[s,a] = self.Qs[s,a] + self.lr*(r + self.gamma*np.max(self.Qs[s_next,a_next]) - self.Qs[s,a])
        
        if self.epsilon > self.epsilon_min:
            self.epsilon*=self.epsilon_decay
    
    def available_actions(self, s):
        #assuming s is a integer
        actions = np.array([i for (i,j) in enumerate(self.env.costs[s,]) if j > 0])
        return actions
    
    def act(self, s):
        actions = self.available_actions(s)  
        non_action = list(set([i for i in range(self.num_action) if i not in actions] + self.state_memmory))

        #Avoid 1: used switchs, 2: not connected
        qs = np.copy(self.Qs[s,:]) 
        qs[non_action] = -np.inf
        
        final_actions = [x for x in actions if x not in non_action]
        
        if len(final_actions) > 0:            
            if np.random.rand() > self.epsilon:
                a = np.argmax(qs)
            else:
                a = np.random.choice(final_actions)
        else:
            #print('no action available')
            return None
        
        return a
    
    def reset_memmory(self):
        self.state_memmory=[]
        
    def remember_state(self, s):
        self.state_memmory.append(s)
        

class TainAgent():
    def __init__(self, q_agent, env, maxm_iters):
        self.q_agent = q_agent  # Has Q table
        self.env = env          # Has cost table
        self.maxm_iters = maxm_iters
        
    def plot_learning_curve(self,x, scores, figure_file='test'):
        running_avg = np.zeros(len(scores))
        for i in range(len(running_avg)):
            running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
        plt.figure()
        plt.plot(x, running_avg)
        plt.title('Running average of previous 100 scores')
        #plt.savefig(figure_file)
    
    
    def run_src_dest(self, src, dest):
        #run the path through src to dest
        if src == dest:
            print('src and dest are the same, No need to run')
            return 
        
        reward_list = []
        print('src: ', src, ' dest: ', dest)
        
        for i in range(self.maxm_iters):
            self.q_agent.reset_memmory()
            total_reward = 0
            s = src

            while s!=dest:
                self.env.state = s
                self.q_agent.remember_state(s)
                a = self.q_agent.act(s)
                if a == None:
                    #print(i)
                    break
                else:
                    s_next, r, done = self.env.step(a)
                    self.q_agent.update(s,a,r,s_next)

                    total_reward+=r
                    s=s_next
            
            reward_list.append(total_reward)
        return reward_list

    
    def find_path(self,src, dest, optim=True):

        self.q_agent.reset_memmory()
        
        hist = []
        cost = 0.
        s = src
        
        if src == dest:
            print('Src and dest are same')
            return 
        
        while s != dest:
            self.q_agent.remember_state(s)
            self.env.state = s
            hist.append(s)
            
            a = self.q_agent.act(s)
            
            if a == None:
                print('No path found')
                break
            else:
                s_next, r, done = self.env.step(a)
                cost+=self.env.costs[s,a]
                s=s_next
        hist.append(dest)
        
        return hist,1.*cost


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
                                   
            
            
            

            
            
            
            
            
        
    