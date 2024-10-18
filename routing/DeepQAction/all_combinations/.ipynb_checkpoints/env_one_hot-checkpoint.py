import gym
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
            
class CitizenBankEnv(gym.Env):
    def __init__(self):
        df = pd.read_csv('../logical_links.csv')
        
        #self.action_space = gym.Env.space.discrete(self.num_action)
        #self.observation_space = gym.Env.space.discrete(self.num_state)
        
        df_device1 = df[['DeviceA']].drop_duplicates().rename(columns={'DeviceA':'DeviceName'})
        df_device2 = df[['DeviceBName']].drop_duplicates().rename(columns={'DeviceBName':'DeviceName'})
        
        self.df_device = pd.concat([df_device1,df_device2]).drop_duplicates().reset_index(drop=True).reset_index(drop=False)
        
        self.df = df.merge(self.df_device, how='inner', left_on=['DeviceA'], right_on='DeviceName').rename(columns={'index':'DeviceA_Code'}).drop('DeviceName', axis=1)
        self.df = self.df.merge(self.df_device,how='inner', left_on='DeviceBName', right_on='DeviceName').rename(columns={'index':'DeviceB_Code'}).drop('DeviceName', axis=1)
 
        self.num_devices = self.df_device.shape[0]
        self.df_device.to_csv("devices.csv", index=False)
        self.df.to_csv('links.csv', index=False)
        self.build_costs()
        self.topology_graph()
        self.reset(0,0)
    
    #Assuming the weight are the same
    def topology_graph(self):
        """Build a graph for the topology."""
        edges = []
        self.graph = defaultdict(list)
        for link in self.df[['DeviceA_Code', 'DeviceB_Code']].values:
            start = link[0]
            end = link[1]
            edges.append([start, end])
            
        for edge in edges:
            first,second = edge[0], edge[1]
            self.graph[first].append(second)
            self.graph[second].append(first)
            
        return
    
    def build_costs(self):
        self.costs = np.zeros((self.num_devices,self.num_devices))
        values = self.df['Bandwidth (Mbps)'].values
        index = self.df[['DeviceA_Code','DeviceB_Code']].values
        self.costs[index[:,0], index[:,1]] = values
        self.costs[index[:,1], index[:,0]] = values
     
    def reset(self, src, dest):
        self.state = np.array([src,dest])
        self.dest = dest
        return self.state       
    
    def step(self, action):
        ############################
        # State includes the [src, dest]
        # Actions is only the next position
        done=False  
        reward = -self.costs[self.state[0], action]
        
        if action == self.dest:
            done=True
            reward = 0        
        self.state = [action, self.dest]
        
        return self.state, reward, done, '_'     
        
    
    def render(self):
        pass
    
    
            