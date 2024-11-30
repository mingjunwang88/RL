import gymnasium as gym
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import networkx as nx
            
class CitizenBankEnv(gym.Env):
    def __init__(self, dest):
        np.random.seed(0)
        self.device = pd.read_csv('devices.csv')
        self.df = pd.read_csv('links.csv')
        self.df['Bandwidth (Mbps)'] = self.df['Bandwidth (Mbps)'].values * np.random.rand(self.df.shape[0]) #added for some randomnessMJ
        self.num_devices = self.device.shape[0]
        self.build_costs()
        self.topology_graph()
        self.dest = dest
        self.reset()
        self.create_graph()
        self.state_zeros = np.zeros(self.num_devices)
    
    def create_graph(self, weight=True):
        self.G=nx.Graph()
        if weight:
            points_list = [(i,j,k) for (i,j,k) in self.df[['DeviceA_Code','DeviceB_Code','Bandwidth (Mbps)']].values]
            self.G.add_weighted_edges_from(points_list)
            edge_labels = nx.get_edge_attributes(self.G, "weight")
        else:
            points_list = [(i,j) for (i,j,k) in self.df[['DeviceA_Code','DeviceB_Code','Bandwidth (Mbps)']].values]
            self.G.add_edges_from(points_list)
        
        pos = nx.spring_layout(self.G, seed=0)
        
        nx.draw_networkx_nodes(self.G,pos)
        nx.draw_networkx_edges(self.G,pos)
        if weight:
            nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels)
        else:
            nx.draw_networkx_edge_labels(self.G, pos,)
        plt.show()
        return 

    
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
     
    def reset(self):
        """ 
        state is just the current position
        Reset a random starting position. The dest did not change
        """
        state = np.random.choice(range(self.num_devices))
        self.state = state
        return self.state       
    
    def step(self, action):
        ############################
        # State is the current position
        # Actions is only the next position
        done=False
        
        if action == self.dest:
            done=True
            
        reward = -self.costs[self.state, action]
        
        self.state = action        
        
        return self.state, reward, done, False, {}    
        
    
    def render(self):
        pass
    
    
            