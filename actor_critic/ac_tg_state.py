CUDA_LAUNCH_BLOCKING=1
import torch as T
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import matplotlib.pyplot as plt
from torch.distributions import Categorical
import numpy as np

###################################################
# Actor Critic mmodel with discrete actions
###################################################

class Critic(nn.Module):
    def __init__(self, dim_state ,h1, h2, alpha=1e-4 ):
        super(Critic, self).__init__()
        
        self.dim_state = dim_state
        self.dim_action = dim_action
        
        self.linear1 = nn.Linear(dim_state, h1)
        self.linear2 = nn.Linear(h1, h2)
        self.linear3 = nn.Linear(h2,1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha, weight_decay=1e-4)
        self.device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, state):
        output = state.to(self.device)
        output = F.relu(self.linear1(output))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        
        return output
    

class Actor(nn.Module):
    def __init__(self, dim_state, dim_action, alpha, fc1, fc2,):
        super(Actor, self).__init__()
        self.lr = alpha
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.fc1 = fc1
        self.fc2 = fc2
        
        self.linear1 = nn.Linear(dim_state, fc1)
        self.linear2 = nn.Linear(fc1, fc2)
        self.linear3 = nn.Linear(fc2, dim_action)
        self.device = T.device('cpu' if T.cuda.is_available() else 'cpu')
        self.optimizer = optim.Adam(self.parameters(), lr=alpha, weight_decay=1e-4)
        self.to(self.device)
    
    def forward(self,state):
        output = state.to(self.device)
        output = F.relu(self.linear1(output))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        
        return output 
        
class Agent():
    def __init__(self, env, dim_state, dim_action, alpha, fc1, fc2, maxm_Iters=50, gamma=0.99):
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.gamma = gamma
        self.maxm_Iters = maxm_Iters
        
        self.device = T.device('cpu' if T.cuda.is_available() else 'cpu')
        self.critic = Critic(dim_state,fc1, fc2)
        self.actor = Actor(dim_state, dim_action, alpha, fc1, fc2)
        
    def plot_learning_curve(self, x, scores, figure_file='test'):
        running_avg = np.zeros(len(scores))
        for i in range(len(running_avg)):
            running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
        plt.plot(x, running_avg)
        plt.title('Running average of previous 100 scores')
        plt.savefig('ac_state.png')
    
    def train(self):
      
        reward_list = []
        steps = 0 
        for i in range(self.maxm_Iters):
            state, _ = env.reset()
            total_reward = 0
            
            done = False
            while not done:
                steps+=1
                self.critic.zero_grad()
                self.actor.zero_grad()
              
                probs = F.softmax(self.actor(T.tensor(state).float()), dim=0)

                m = Categorical(probs)
                action = m.sample().to(self.device)

                state_next, reward, terminate, trunc, _ = env.step(action.item())
                done = np.logical_or(terminate, trunc)
                total_reward +=reward

                target = torch.tensor(reward) + self.gamma * self.critic(T.tensor(state_next).float())*(1-int(done))
                state_value = self.critic(T.tensor(state).float())
                advantage = target - state_value
         
                ## Loss for critic
                loss_critic = advantage**2
                
                ## Loss for actor
                loss_actor = -m.log_prob(action) * advantage
                
                (loss_critic + loss_actor).backward()
                
                self.actor.optimizer.step()
                self.critic.optimizer.step()
                
                state = state_next
            if i % 100 == 0:
                print('episode: ', i, ' total reward: ',total_reward )
            reward_list.append(total_reward)
            
        x = [i+1 for i in range(self.maxm_Iters)]
        self.plot_learning_curve(x, reward_list)
        
if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    ##env = gym.make("MountainCar-v0")
    #env = gym.make('Acrobot-v1')
    dim_state = env.observation_space.shape[0]
    dim_action = env.action_space.n
    print (dim_state,dim_action )
    
    agent = Agent(env,dim_state,dim_action, 1e-5, 128, 128, maxm_Iters=2000)
    agent.train()
