import gymnasium as gym
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import networkx as nx
            
class CitizenBankEnv(gym.Env):
    def __init__old(self):
        df = pd.read_csv('../logical_links.csv')        
        df_device1 = df[['DeviceA']].drop_duplicates().rename(columns={'DeviceA':'DeviceName'})
        df_device2 = df[['DeviceBName']].drop_duplicates().rename(columns={'DeviceBName':'DeviceName'})
        
        self.df_device = pd.concat([df_device1,df_device2]).drop_duplicates().reset_index(drop=True).reset_index(drop=False)
        
        self.df = df.merge(self.df_device, how='inner', left_on=['DeviceA'], right_on='DeviceName').rename(columns={'index':'DeviceA_Code'}).drop('DeviceName', axis=1)
        self.df = self.df.merge(self.df_device,how='inner', left_on='DeviceBName', right_on='DeviceName').rename(columns={'index':'DeviceB_Code'}).drop('DeviceName', axis=1)
        
        self.df['cost'] = 1.0 / self.df['Bandwidth (Mbps)'] * np.random.rand(self.df.shape[0])
        
        self.num_devices = self.df_device.shape[0]
        self.df_device.to_csv("devices.csv", index=False)
        self.df.to_csv('links.csv', index=False)
        self.build_costs()
        self.topology_graph()
        self.reset()

    def __init__(self):
        self.df_device = pd.read_csv("devices.csv",)
        self.df = pd.read_csv('links.csv',)
        self.df['cost'] = self.df['cost'] * 1000
        self.num_devices = self.df_device.shape[0]

        self.build_costs()
        self.topology_graph()
        self.reset()
        self.create_graph()

    def create_graph(self, weight=True):
        self.G=nx.Graph()
        if weight:
            points_list = [(i,j,k) for (i,j,k) in self.df[['DeviceA_Code','DeviceB_Code','cost']].values]
            self.G.add_weighted_edges_from(points_list)
            edge_labels = nx.get_edge_attributes(self.G, "weight")
        else:
            points_list = [(i,j) for (i,j,k) in self.df[['DeviceA_Code','DeviceB_Code','cost']].values]
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
        values = self.df['cost'].values
        index = self.df[['DeviceA_Code','DeviceB_Code']].values
        self.costs[index[:,0], index[:,1]] = values
        self.costs[index[:,1], index[:,0]] = values
     
    def reset(self):
        #state includes the current position
        state = np.random.choice(range(self.num_devices), 2)
        self.state = state[0]
        self.dest = state[1]
        
        #self.state = 3
        #self.dest = 26
        return self.state       
    
    def step(self, action):
        ############################
        # State includes the [src, dest]
        # Actions is only the position
        done=False
        
        if action == self.dest:
            done=True
            
        reward = -self.costs[self.state, action]
        
        self.state = action
        
        return self.state, reward, done, False, {}     
        
    
    def render(self):
        pass
    
    
            