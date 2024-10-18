import torch.nn as nn
import torch 
import torch as T
import torch.optim as optim
import torch.nn.functional as F
import gym
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
from itertools import permutations, product, combinations

#################################################
# Model output output Q(s,a) for one single value
#################################################

class Model(nn.Module):
    def __init__(self, num_state, h1, h2, alpha=1e-4):
        ##########################################
        # 1: Each node is represented by a one hot vector
        # 2: Only topology spesfic, not generlaize to other topologies.
        # 3: Assume number of states an actions are same(number of nodes)
        # 4: State = onehot(src) + onehot(dest)
        # 5: Action = onehot(node)
        # 6: num_state: number of nodes
        ##########################################
        super(Model, self).__init__()        
        self.linear1 = nn.Linear(3*num_state, h1)
        self.linear2 = nn.Linear(h1, h2)
        self.linear3 = nn.Linear(h2,1)
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha, weight_decay=1e-4)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def forward(self, state, action):
        ####################################################
        # State is represented by concatnate of src and dest
        ####################################################
        state = state.to(self.device)
        action = action.to(self.device)
        
        output = T.cat((state, action), dim=1)
        
        output = F.relu(self.linear1(output))
        output = F.relu(self.linear2(output))
        output = self.linear3(output)
        return output
    
    def save_CheckPoint(self):
        print ('...save check point...')
        torch.save(self.state_dict, self.checkpoint_file)
        
    def load_CheckPOint(self):
        print ('...load check point...')
        self.load_state_dict(torch.load(self.checkpoint_file))
      
    
class replayBuffer():
    def __init__(self,max_memsize, dim_state, dim_action):
        self.max_memsize = max_memsize
        self.counter = 0
        
        self.states_mem = T.zeros((max_memsize, dim_state+dim_state), dtype=T.float32)
        self.states_mem_new = T.zeros((max_memsize, dim_state+dim_state), dtype=T.float32)
        self.action_mem = T.zeros((max_memsize,dim_action), dtype=T.float32)
        self.reward_mem = np.zeros(max_memsize, dtype=np.float32)
        self.done_mem = np.zeros(max_memsize, dtype=bool) 
        self.action_next_mem = T.zeros((max_memsize,dim_action), dtype=T.float32)
        
    def store_transaction(self, state, action, reward, state_new, actions_new, done):
        i = self.counter % self.max_memsize
        
        self.states_mem[i] = state
        self.action_mem[i] = action
        self.reward_mem[i] = reward
        self.done_mem[i]  =done
        self.states_mem_new[i] = state_new
        self.action_next_mem[i] = actions_new
        
        self.counter+=1
           
    def sample_buffer(self, batch_size):
        max_mem = min(self.counter, self.max_memsize)
        
        batch = np.random.choice(max_mem, batch_size, replace=False)

        state = self.states_mem[batch]
        action = self.action_mem[batch]
        reward = self.reward_mem[batch]
        state_new = self.states_mem_new[batch]
        action_new = self.action_next_mem[batch]
        done = self.done_mem[batch]
        return state, action, reward, state_new, action_new, done


class Agent():
    def __init__(self, env, num_state, h1, h2, max_memsize, alpha=1e-4, batch_size=258, max_iters = 30, epsilon=0.05, steps_target=1000, gamma=0.99,  \
                min_epsilon=0.005): 
        self.batch_size = batch_size
        self.gamma = gamma
        self.max_iters = max_iters
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_memsize = max_memsize
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.buffer = replayBuffer(max_memsize,num_state, num_state)
        self.loss_fun = nn.MSELoss()

        self.model = Model(num_state,h1, h2, alpha=alpha).to(self.device)
        self.model_target = Model(num_state, h1, h2, alpha=alpha).to(self.device)
        self.steps_target = steps_target
        self.env = env 
        self.blank = T.zeros(num_state)
        self.num_state = num_state
        
    def plot_learning_curve(self,x, scores, figure_file='test'):
        running_avg = np.zeros(len(scores))
        for i in range(len(running_avg)):
            running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
        plt.figure()
        plt.plot(x, running_avg)
        plt.title('Running average of previous 100 scores')
        #plt.savefig(figure_file)
    
    def reset_memmory(self):
        self.memmory = []
        
    def remember_memmory(self, pos):
        self.memmory.append(pos)
        
    def available_actions(self, current_pos):
        actions = [i for i, j in enumerate(self.env.costs[current_pos,]) if j > 0]
        actions = [i for i in actions if i not in self.memmory]
        return actions
    
    def act(self, current_pos):
        ##################################
        # Return next position as a tensor
        ##################################
        actions = self.available_actions(current_pos)
        n = len(actions)
        
        if n == 0:
            return None
        
        if np.random.rand() < self.epsilon:
            return T.tensor(np.random.choice(actions)).to(self.device)
        else:
            # One hot encoding src, dest and action            
            src = copy(self.blank)
            dest = copy(self.blank)
            
            src[current_pos] = 1
            dest[self.env.dest] = 1

            src = src.repeat(n,1)
            dest =dest.repeat(n,1)        
            inpt = T.cat((src,dest), dim=1)
            
            actions_tensor = T.zeros(n, self.env.num_devices)
            actions_tensor[range(n),actions] =1  
            
            Qs = self.model(inpt,actions_tensor)
            
            if len(Qs)>0:
                best = T.argmax(Qs)
                return T.tensor(actions[best])
            return None
        
    def best_action(self, current_pos):
        ##################################
        # Return next position as a tensor
        ##################################
        actions = self.available_actions(current_pos)
        n = len(actions)
        
        if n == 0:
            return None
        # One hot encoding src, dest and action            
        src = copy(self.blank)
        dest = copy(self.blank)
          
        src[current_pos] = 1
        dest[self.env.dest] = 1

        src = src.repeat(n,1)
        dest =dest.repeat(n,1)        
        inpt = T.cat((src,dest), dim=1)
            
        actions_tensor = T.zeros(n, self.env.num_devices)
        actions_tensor[range(n),actions] =1  
            
        Qs = self.model(inpt,actions_tensor)
        if len(Qs)>0:
            best = T.argmax(Qs)
            return T.tensor(actions[best])
        return None
    
    def one_hot(self, dim, idx):
        x = T.zeros(dim)
        x[idx] = 1
        return x
         
    def train(self, src, dest):
        index = range(self.batch_size)
        self.model.train()
        self.model_target.eval()
        rewards_list = []
        its = 1
        
        #set state first
        state = self.env.reset(src, dest)
        print('initial state: ', state)
        
        for i in range(self.max_iters):
            self.reset_memmory()
            done=False
            rewards = 0
            its+=0
            len_episod = 0

            while not done:
                self.model.zero_grad()
                self.remember_memmory(state)
                len_episod+=1
                action = self.act(state) #return a tensor 
                if action == None:
                    break

                state_new, reward, done, _ = self.env.step(action.item())
                rewards+=reward

                if not done:
                    action_new = self.act(state_new)
                    if action_new == None:
                        break
                else:
                    action_new= action #Does not affect since it will be timed by zero

                state_embd = T.cat((self.one_hot(self.num_state, state[0]),self.one_hot(self.num_state, state[1])))
                action_embed = self.one_hot(self.num_state, action.item())
                state_new_embed = T.cat(( self.one_hot(self.num_state, state_new[0]),self.one_hot(self.num_state, state_new[1])))
                action_new_embd = self.one_hot(self.num_state, action_new.item())

                self.buffer.store_transaction(state_embd, action_embed, reward, state_new_embed, action_new_embd, done)

                if self.buffer.counter > self.batch_size:
                    state_batch, action_batch, reward_batch, state_new_batch, action_next_batch, done_batch = self.buffer.sample_buffer(self.batch_size)

                    # Q value
                    q_values = T.squeeze(self.model(state_batch,action_batch))

                    # Target value
                    with T.no_grad():              
                        q_targets = T.squeeze(self.model_target(state_new_batch, action_next_batch))
                        q_targets[done_batch] = 0.0                  
                        q_targets = torch.tensor(reward_batch).to(self.device) + self.gamma * q_targets

                    # Loss function
                    loss = self.loss_fun(q_targets,q_values)
                    loss.backward(retain_graph=True)

                    # Update weights
                    self.model.optimizer.step()

                    #Update the state
                    state = state_new
                    if (self.buffer.counter % self.steps_target) == 0:
                        self.model_target.load_state_dict(self.model.state_dict())

                    if self.epsilon > self.min_epsilon:
                        self.epsilon=self.epsilon/its

            rewards_list.append(rewards)
        x = [i+1 for i in range(self.max_iters)]  
        self.plot_learning_curve(x, rewards_list)

    def train_combinations(self):
        index = range(self.batch_size)
        self.model.train()
        self.model_target.eval()
        rewards_list = []
                
        for i in range(self.max_iters):
            for k in combinations(range(self.num_state), r=2):
                its = 0
                self.reset_memmory()
                state = self.env.reset(k[0], k[1])
                done=False
                rewards = 0
                len_episod = 0

                while not done:
                    its+=1
                    
                    if its > 1000:
                        break
                        
                    self.model.zero_grad()
                    self.remember_memmory(state[0])
                    len_episod+=1

                    action = self.act(state[0]) #return a tensor 
                    if action == None:
                        break

                    state_new, reward, done, _ = self.env.step(action.item())
                    rewards+=reward

                    if not done:
                        action_new = self.act(state_new[0])
                        if action_new == None:
                            break
                    else:
                        action_new= action #Does not affect since it will be timed by zero

                    state_embd = T.cat((self.one_hot(self.num_state, state[0]),self.one_hot(self.num_state, state[1])))
                    action_embed = self.one_hot(self.num_state, action.item())
                    state_new_embed = T.cat(( self.one_hot(self.num_state, state_new[0]),self.one_hot(self.num_state, state_new[1])))
                    action_new_embd = self.one_hot(self.num_state, action.item())

                    self.buffer.store_transaction(state_embd, action_embed, reward, state_new_embed, action_new_embd, done)

                    if self.buffer.counter > self.batch_size:
                        state_batch, action_batch, reward_batch, state_new_batch, action_next_batch, done_batch = self.buffer.sample_buffer(self.batch_size)

                        # Q value
                        q_values = T.squeeze(self.model(state_batch,action_batch))

                        # Target value
                        with T.no_grad():              
                            q_targets = T.squeeze(self.model_target(state_new_batch, action_next_batch))
                            q_targets[done_batch] = 0.0                  
                            q_targets = torch.tensor(reward_batch).to(self.device) + self.gamma * q_targets

                        # Loss function
                        loss = self.loss_fun(q_targets,q_values)
                        loss.backward(retain_graph=True)

                        # Update weights
                        self.model.optimizer.step()

                        #Update the state
                        state = state_new
                        if (self.buffer.counter % self.steps_target) == 0:
                            self.model_target.load_state_dict(self.model.state_dict())

                        if self.epsilon > self.min_epsilon:
                            self.epsilon=self.epsilon/its

                rewards_list.append(rewards)
        num_combs = len([i for i in combinations(range(self.num_state),r=2)])
        x = [i+1 for i in range(self.max_iters*num_combs)]  
        self.plot_learning_curve(x, rewards_list)
        
    def find_path(self,src, dest):
        self.reset_memmory()
        self.env.dest = dest
        
        hist = []
        cost = 0.
        s = src
        its = 0
        
        if src == dest:
            print('Src and dest are same')
            return 
        
        while s != dest:
            its+=1
            
            if its > 100:
                break            
            self.remember_memmory(s)
            hist.append(s)
            a = self.best_action(s)  
            if a == None:
                print('No path found')
                break
            else:
                s_next, r, done, _ = self.env.step(a.item())
                cost+=self.env.costs[s,a]
                s=s_next[0]
                print(a.item())
                
        hist.append(dest)
        
        return hist,1.*cost
        

for i in [100, 1000,]:
    
    if __name__ == '__main__':
        #env = gym.make('AirRaid-ram-v0')
        env = gym.make('CartPole-v1')
        #env = gym.make('MountainCar-v0')
        dim_state = env.observation_space.shape[0]
        dim_action = env.action_space.n
        print (dim_state ,dim_action)
        #env, dim_state, dim_action,h1, h2, max_memsize, alpha=1e-4, batch_size=258, max_iters = 30, epslon=0.05, steps_target=1000, gamma=0.99):
        agent = Agent(env,dim_state,dim_action, 256, 256,max_memsize=1000000, max_iters=500,steps_target=1)
        agent.train()
        