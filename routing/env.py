import gym
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

class DeliveryEnv(gym.Env):
    def __init__(self, max_box, stops, method,render_each=100):
        self.n_stops = stops
        self.max_box = max_box
        self.method = method
        self._generate_stops()
        self._generate_costs()
        self.reset()
        self.render()
        self.render_each = render_each

    def _generate_stops(self):
        #Generate the stops with (x,y)
        self.xy = np.random.rand(self.n_stops, 2) * self.max_box
        self.x = self.xy[:,0]
        self.y = self.xy[:,1]
        return
        
    def _generate_costs(self):
        #Provide the cost(R) matrix among the stops, can be distiance, time etc.
        if self.method in ("distance","traffic_box"):
            self.cost = cdist(self.xy, self.xy, 'euclidean')
        elif self.method in ('time'):
                self.cost = np.random.rand(self.n_stops, self.n_stops) * self.max_box
                np.fill_diagonal(self.cost, 0)
        else:
            print('Need to use the correct method!')
        return 
    
    def reset(self):
        #set the initial stop and return the current stop which is an integer
        self.stops = []
        
        first_stop = np.random.randint(self.n_stops)
        self.stops.append(first_stop)
        
        return first_stop
    
    def step(self, action):
        
        r = self.cost[self.stops[-1], action] 
        self.stops.append(action)
        
        done =False
        if len(self.stops) == self.n_stops:
            done= True
        
        #action is actually the next state
        return action, r, done
    
    def render(self):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.set_title('Stops')
        ax.scatter(self.x, self.y)
        
        #Show start
        if len(self.stops)>0:
            idx = self.stops[0]
            xy = (self.x[idx],self.y[idx])
            
            xytext = (xy[0]+0.1,xy[1]-0.05)
            ax.annotate("START",xy=xy,xytext=xytext,weight = "bold")
            
        if len(self.stops) > 1:
            idx = self.stops[-1]
            xy = (self.x[idx],self.y[idx])            
            xytext = (xy[0]+0.1,xy[1]-0.05)
            ax.annotate("END",xy=xy,xytext=xytext,weight = "bold")
            
            idx_x = self.x[self.stops]
            idx_y = self.y[self.stops]
            ax.plot(idx_x, idx_y)
            
            
            
            
            
            
    
            
    
        
        
        
