import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time 
from copy import copy

class DP_PlanningBase():
    def __init__(self, env, delta=0.1, alpha=0.99):
        self.env = env
        self.V = np.zeros(env.num_devices)
        self.delta = delta
        self.alpha = alpha
    
    def avaiable_actions(self, position):
        actions = [i for i, j in enumerate(self.env.costs[position,]) if j > 0]
        return actions
        
    def value_iteration(self, dest, value=1):
        diff = self.delta
        costs = copy(self.env.costs)    
        
        for i, j in enumerate(costs[:,dest]):
            if j > 0:
                costs[int(i),int(dest)] = value #make the dest have larger reward!!
        
        diffs = np.zeros(self.env.num_devices)
        self.PI = np.zeros(self.env.num_devices)
        
        j=0
        while diff >= self.delta:
            for s in range(len(self.V)):
                j+=1

                actions = self.avaiable_actions(s)
                v_ = copy(self.V[s])
                
                #Since the next step is the same as the action with probability of 1, We use a simple fomula
                if s != dest:
                    Qs = -costs[s,:][actions] + self.alpha*self.V[actions]
                    self.V[s] = np.max(Qs)
                
                diffs[s] = abs(v_- self.V[s])
            diff = np.sum(diffs)
 
        #Return the optimal policy
        for s in range(len(self.V)):
            actions = self.avaiable_actions(s)
            if s != dest:
                Qs = -costs[s,:][actions] + self.alpha*self.V[actions]
                idx = np.argmax(Qs)
                self.PI[s] = actions[idx]
            else:
                self.PI[s] = dest
        
        return self.V, self.PI
            
    def find_path(self, src, dest):
        s = int(copy(src))
        path = []
        
        if src == dest:
            print('Already in the dest')
            path.append(dest)
            return path
        
        while s != dest:
            path.append(int(s))
            s = self.PI[int(s)]
        path.append(dest)
        
        return path            
    
    def policy_eval(self,):
        pass
    
    def policy_improv(self,):
        pass
    
    def policy_iteration():
        self.policy_eval()
        self.pllicy_improv()
        
if __name__ == '__main__':
    planner = DP_PlanningBase(env)
    
    start = time.time()
    for dest in [0,1,2,3,4]:
        v, p = planner.value_iteration(dest)
        for src in range(34): 
            print('DP path:          ', planner.find_path(src, dest))
            print('dijkstra path:    ', nx.shortest_path(G, src, dest, weight='Bandwidth (Mbps)'))
        print()
    print('---- total ---:', time.time() - start)