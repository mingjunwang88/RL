import importlib
import AgentBase
importlib.reload(AgentBase)
from AgentBase import DP_PlanningBase    

class PolicyIteration(DP_PlanningBase):
    def __init__(self, env, delta=0.1, alpha=0.95):
        self.env = env
        self.V = np.zeros(env.num_devices)
        self.delta = delta
        self.alpha = alpha
        self.generate_random_policy()
        self.holder = np.zeros(env.num_devices)
    
    def generate_random_policy(self):
        ################################
        #generate initial random policy
        ################################
        self.PI = np.zeros(env.num_devices).astype(int)
        for s in range(self.env.num_devices):
            actions = self.available_actions(s)
            self.PI[s] = np.random.choice(actions)
        return 
    
    def available_actions(self, s):
        actions = [i for i, j in enumerate(self.env.costs[s,]) if j > 0]
        return actions
    
    def set_costs(self, dest, value):
        ###########################################
        # re adjust the cost function for each dest
        ###########################################
        costs = copy(self.env.costs)
        for m,n in enumerate(costs[:,dest]):
            if n > 0:
                costs[m,dest] = value #make the dest have larger reward!!
        return costs
        
    def policy_improv(self, dest, value=1):
        ##################################
        # Provide the updated self.PI
        ##################################
        pi_ = copy(self.PI)
        costs = self.set_costs(dest, value)        
        
        for s in range(self.env.num_devices):
            if s != dest:
                actions = self.available_actions(s) 
                Qs = -costs[s][actions] + self.alpha*self.V[actions]    
                idx = np.argmax(Qs)
                self.PI[s] = actions[idx]
            else:
                self.PI[s] = dest
        stable =np.sum(abs(pi_ - self.PI))

        return stable
        
    def policy_eval(self, dest, value=1):
        ###########################################
        # Provide updated the self.V after each run
        ###########################################  
        diff = self.delta
        costs = self.set_costs(dest, value) 
        
        i=0
        while diff >= self.delta:
            i+=1
            for s in range(self.env.num_devices):
                v_ = copy(self.V[s])
                if s != dest:
                    self.V[s] = -costs[s,self.PI[s]] + self.alpha*self.V[self.PI[s]]  
                else:
                    self.V[s] = -costs[s,self.PI[s]]
                self.holder[s] = np.abs(v_ - self.V[s])
            diff = np.sum(self.holder)
            #print(diff)
        return 

    def policy_iteration(self, dest, value=1):
        stable = True
        
        i=0
        while stable:
            i+=1
            self.policy_eval(dest)
            stable = self.policy_improv(dest)
            if i % 1 == 0:
                #print('iterations: ', i, stable)
                pass
        return 
        
planner = PolicyIteration(env)
#planner.policy_eval(23)
#print(planner.PI)
#planner.policy_improv(23)
#print(planner.PI)
#print(planner.V)
#print(planner.PI)

#src = 13
#dest = 0
#planner.policy_iteration(dest)
#print('V: ', planner.V)
#print('PI: ', planner.PI)
#print('DP:            ', planner.find_path(src, dest))
#print('dijkstra path: ', nx.shortest_path(G, src, dest, weight='Bandwidth (Mbps)'))

start = time.time()
for dest in range(env.num_devices):
    #print('destination: ', dest)
    planner.policy_iteration(dest)
    for src in range(34): 
        #print('DP path:          ', planner.find_path(src, dest))
        #print('dijkstra path:    ', nx.shortest_path(G, src, dest, weight='Bandwidth (Mbps)'))
        #planner.find_path(src, dest)
        pass
#print('---- total ---:', time.time() - start)
