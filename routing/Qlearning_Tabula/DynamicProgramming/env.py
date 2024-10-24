import gymnasium as gym
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from scipy import sparse
import networkx as nx
import matplotlib.pyplot as plt

class CitizenBankEnv(gym.Env):
    def __init__old(self, dest):
        df = pd.read_csv('../../logical_links.csv')
        
        #self.action_space = gym.Env.space.discrete(self.num_action)
        #self.observation_space = gym.Env.space.discrete(self.num_state)
        
        df_device1 = df[['DeviceA']].drop_duplicates().rename(columns={'DeviceA':'DeviceName'})
        df_device2 = df[['DeviceBName']].drop_duplicates().rename(columns={'DeviceBName':'DeviceName'})
        
        self.df_device = pd.concat([df_device1,df_device2]).drop_duplicates().reset_index(drop=True).reset_index(drop=False)
        
        self.df = df.merge(self.df_device, how='inner', left_on=['DeviceA'], right_on='DeviceName').rename(columns={'index':'DeviceA_Code'}).drop('DeviceName', axis=1)
        self.df = self.df.merge(self.df_device,how='inner', left_on='DeviceBName', right_on='DeviceName').rename(columns={'index':'DeviceB_Code'}).drop('DeviceName', axis=1)
 
        self.num_devices = self.df_device.shape[0]
        self.build_costs()
        self.topology_graph()
        self.dest = dest
        self.reset(0)

    def __init__(self, dest):
        np.random.seed(0)
        self.device = pd.read_csv('devices.csv')
        self.df = pd.read_csv('links.csv')
        self.df['Bandwidth (Mbps)'] = self.df['Bandwidth (Mbps)'].values * np.random.rand(self.df.shape[0]) #added for some randomnessMJ
        self.num_devices = self.device.shape[0]
        self.build_costs()
        self.topology_graph()
        self.dest = dest
        self.reset(0)
        self.create_graph()
    
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
    
    def reset_memmory(self):
        self.memmory = []
        
    def remember_memmory(self, pos):
        self.memmory.append(pos)
        
    def build_costs(self):
        """ 
        Build a dense cost matrix
        """
        self.costs = np.zeros((self.num_devices,self.num_devices))
        costs = self.df['Bandwidth (Mbps)'].values #* np.random.rand(self.df.shape[0]) #added for some randomnessMJ
        index = self.df[['DeviceA_Code','DeviceB_Code']].values
        self.costs[index[:,0], index[:,1]] = costs
        self.costs[index[:,1], index[:,0]] = costs
        self.cost_sparse = sparse.coo_matrix(self.costs)
     
    def reset(self, src):
        #########################
        # The dest is fixed. There can be any src
        # This can reduce the number of models comparing to using src/dest
        # state is the current location
        #######################
        
        #self.state = np.random.choice(self.num_devices)
        self.state = src
        self.reset_memmory()
        self.remember_memmory(self.state)
        
        return self.state
    
    def step(self, action):
        ############################
        # Actions is only the next position
        done=False
        reward = -self.costs[self.state, action]
        
        """
        if action in self.memmory:
            done = True
            reward = -2000.
        """    
        if action == self.dest:
            done=True
            reward = 0         
        
        self.state = action
        self.remember_memmory(action)

        
        return self.state, reward, done, True, '_'     
        
    
    def render(self):
        pass
    
    
            