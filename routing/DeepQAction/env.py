import gymnasium as gym
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import networkx as nx
from torch_geometric.utils import from_networkx
import torch
            
class CitizenBankEnv(gym.Env):
    def __init__(self):
        np.random.seed(0)
        self.device = pd.read_csv('devices.csv')
        self.df = pd.read_csv('links.csv')
        self.df['Bandwidth (Mbps)'] = self.df['Bandwidth (Mbps)'].values * np.random.rand(self.df.shape[0]) #added for some randomnessMJ
        self.num_devices = self.device.shape[0]
        self.build_costs()
        self.topology_graph()
        self.create_graph()
        self.state_zeros = np.zeros(self.num_devices)
        self.reset()
    
    #Assuming the weight are the same
    def topology_graph(self):
        """Build a graph for the topology as dictonary"""
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
    
    def build_graph(self):
        """Build a networkx graph"""
        points_list = [(i,j) for i,j in self.df[['DeviceA_Code','DeviceB_Code']].values]
        G = nx.Graph()
        G.add_edges_from(points_list)
        self.graph = from_networkx(G)
        if self.graph.num_node_features == 0:
            self.graph.x = torch.ones(self.num_devices, self.num_devices)
    
    def build_costs(self):
        self.costs = np.zeros((self.num_devices,self.num_devices))
        values = self.df['Bandwidth (Mbps)'].values
        index = self.df[['DeviceA_Code','DeviceB_Code']].values
        self.costs[index[:,0], index[:,1]] = values
        self.costs[index[:,1], index[:,0]] = values
     
    def reset(self):
        #state includes the current and dest position
        self.state = np.random.choice(range(self.num_devices), 2, replace=False)
        #self.state = np.array([src,dest])
        self.dest = self.state[1]
        
        return self.state       
    
    def step(self, action):
        ############################
        # State includes the [src, dest]
        # Actions is only the next position
        done=False
        
        if action == self.dest:
            done=True
        
        reward = -self.costs[self.state[0], action]
        self.state = [action, self.dest]
        
        return self.state, reward, done, '_'     
        
    
    def render(self):
        pass
    
    
            