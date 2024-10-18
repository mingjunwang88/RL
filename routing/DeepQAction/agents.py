import torch.nn as nn
import torch 
import torch as T
import torch.optim as optim
import torch.nn.functional as F
import gym
import numpy as np
import matplotlib.pyplot as plt
from itertools import permutations, product, combinations

#################################################
# Model output output Q(s,a) for one single value
#################################################

class Model(nn.Module):
    def __init__(self, num_state, h1, h2, alpha=1e-4, dim_embed=36, one_hot=False):
        ##########################################
        # 1: Each node is represented by a embedding
        # 2: Only topology spesfic, not generlaize to other topologies.
        # 3: Assume number of states an actions are same(number of nodes)
        # 4: State = embed(src) + embed(dest)
        # 5: Action = embed(node)
        # 6: num_state: number of nodes
        ##########################################
        super(Model, self).__init__()
        
        if one_hot:
            dim = num_state
        else:
            dim = dim_embed
            self.embed = nn.Embedding(num_state, dim)
        
        self.linear1 = nn.Linear(3*dim, h1)
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
    def __init__(self, env, num_state, dim_state, dim_action,h1, h2, max_memsize, alpha=1e-4, batch_size=258, max_iters = 30, epsilon=0.05, steps_target=1000, gamma=0.99):
        self.batch_size = batch_size
        self.gamma = gamma
        self.max_iters = max_iters
        self.dim_action = dim_action
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_memsize = max_memsize
        self.epsilon = epsilon
        self.buffer = replayBuffer(max_memsize,dim_state, dim_action)
        self.loss_fun = nn.MSELoss()

        self.model = Model(num_state,h1, h2, alpha=1e-4,dim_embed=dim_state).to(self.device)
        self.model_target = Model(num_state, h1, h2, alpha=1e-4,dim_embed=dim_state).to(self.device)
        self.steps_target = steps_target
        self.env = env 
        
    def plot_learning_curve(self,x, scores, figure_file='test'):
        running_avg = np.zeros(len(scores))
        for i in range(len(running_avg)):
            running_avg[i] = np.mean(scores[max(0, i-50):(i+1)])
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
            src = self.model.embed(T.tensor(current_pos).repeat(n)).to(self.device)
            dest = self.model.embed(T.tensor(self.env.dest).repeat(n)).to(self.device)
            
            actions_tensor = self.model.embed(T.tensor(actions, dtype=T.int)).to(self.device)
            inpt = T.cat((src,dest), dim=1)
            
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
        
        src = self.model.embed(T.tensor(current_pos).repeat(n)).to(self.device)
        dest = self.model.embed(T.tensor(self.env.dest).repeat(n)).to(self.device)
            
        actions_tensor = self.model.embed(T.tensor(actions, dtype=T.int)).to(self.device)
        inpt = T.cat((src,dest), dim=1)
            
        Qs = self.model(inpt,actions_tensor)
        if len(Qs)>0:
            best = T.argmax(Qs)
            return T.tensor(actions[best])
        return None
         
    def train(self):
        index = range(self.batch_size)
        self.model.train()
        self.model_target.eval()
        self.rewards_list = []
        
        for i in range(self.max_iters):
            self.reset_memmory()
            state = self.env.reset()
            done=False
            rewards = 0
            j=0
            while not done:
                j+=1
                self.model.zero_grad()
                self.remember_memmory(state[0])

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

                state_embd = self.model.embed(T.tensor(state)).view(1,-1) #including src and dest
                action_embed = self.model.embed(action)
                state_new_embed = self.model.embed(T.tensor(state_new)).view(1,-1)
                action_new_embd = self.model.embed(action_new)

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
            self.rewards_list.append(rewards)
        
        #num_combs = len([i for i in combinations(range(self.env.num_devices),r=2)])
        x = [i+1 for i in range(self.max_iters)]
        self.plot_learning_curve(x, self.rewards_list)
       
    def train_combinations(self):
        index = range(self.batch_size)
        self.model.train()
        self.model_target.eval()
        self.rewards_list = []
        
        for i in range(self.max_iters):
            for k in combinations(range(self.env.num_devices),r=2):
                self.reset_memmory()
                state = self.env.reset(k[0], k[1])
                done=False
                rewards = 0
                j=0
                while not done:
                    j+=1
                    self.model.zero_grad()
                    self.remember_memmory(state[0])

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

                    state_embd = self.model.embed(T.tensor(state)).view(1,-1) #including src and dest
                    action_embed = self.model.embed(action)
                    state_new_embed = self.model.embed(T.tensor(state_new)).view(1,-1)
                    action_new_embd = self.model.embed(action_new)

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
                self.rewards_list.append(rewards)
        
        num_combs = len([i for i in combinations(range(self.env.num_devices),r=2)])
        x = [i+1 for i in range(self.max_iters*num_combs)]
        self.plot_learning_curve(x, self.rewards_list)
        
    def find_path(self,src, dest):
        self.reset_memmory()
        self.env.dest = dest
        
        hist = []
        cost = 0.
        s = src
        
        if src == dest:
            print('Src and dest are same')
            return 
        
        while s != dest:
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
        