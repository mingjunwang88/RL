import gym
import numpy as np

class PriceEnv(gym.Env):
    
    ## Environment parameters
    T = 20
    price_max = 500
    price_step = 10
    q_0 = 5000
    k = 20
    unit_cost = 100
    a_q = 300
    b_q = 100
    price_grid = np.arange(price_step, price_max, price_step)
    
    
    ##Since for the descrete action 
    
    ## Action size = len(price_grid)
    ## State_size = 2*T
    dim_action = len(price_grid)
    dim_state = 2*T
    
    def __init__(self, config=None):
        self.state = np.repeat(0, 2*self.T)
        
        self.action_space = gym.spaces.Discrete(self.dim_action)
        # NB: Ray throws exceptions for any `0` value Discrete
        # observations so we'll make position a 1's based value
        high = 5000.0* np.ones(self.dim_state)
        low = -high      
        self.observation_space = gym.spaces.Box(low=low, high=high)
        
        self.t = 0
        
        ## new start here
        self.action_space = gym.spaces.
    
    def reset(self):
        self.state = np.repeat(0, 2*self.T)
        self.t = 0
        return self.state
       
    def step_old(self, t, action):
        next_state = np.repeat(0, len(self.state))
        next_state[0] = self.price_grid[action]
        next_state[1:self.T] = self.state[0:self.T-1]
        next_state[self.T+t] = 1
        reward = self.profit_t_response(next_state[0], next_state[1])
        self.state = next_state
        return self.state, reward, 0, 0
    
    def step(self, action):
        done = False
        t = self.t % self.T
        next_state = np.repeat(0, len(self.state))
        next_state[0] = self.price_grid[action]
        next_state[1:self.T] = self.state[0:self.T-1]
        next_state[self.T+t] = 1
        reward = self.profit_t_response(next_state[0], next_state[1])
        self.state = next_state
        
        #print('reward: ', type(reward.ietm()), reward.item())
        self.t+=1
        if self.t % self.T == 0:
            done=True
            
        return (self.state, float(reward), done, {})
    
    def render(self):
        pass
    
    def close(self):
        pass
    
    ## Environment simulator
    def plus(self,x):
        return 0 if x < 0 else x

    def minus(self,x):
        return 0 if x > 0 else -x

    def shock(self,x):
        return np.sqrt(x)

    # Demand at time step t for current price p_t and previous price p_t_1
    def q_t(self,p_t, p_t_1, q_0, k, a, b):
        return self.plus(q_0 - k*p_t - a*self.shock(self.plus(p_t - p_t_1)) + b*self.shock(self.minus(p_t - p_t_1)))

    # Profit at time step t
    def profit_t(self,p_t, p_t_1, q_0, k, a, b, unit_cost):
        return self.q_t(p_t, p_t_1, q_0, k, a, b)*(p_t - self.unit_cost) 

    # Total profit for price vector p over len(p) time steps
    def profit_total(self, p, unit_cost, q_0, k, a, b):
        return self.profit_t(p[0], p[0], q_0, k, 0, 0, unit_cost) + sum(map(lambda t: profit_t(p[t], p[t-1], q_0, k, a, b, unit_cost), range(len(p))))
    
    ## Partial bindings for readability
    def profit_t_response(self, p_t, p_t_1):
        return self.profit_t(p_t, p_t_1, self.q_0, self.k, self.a_q, self.b_q, self.unit_cost)

    def profit_response(self, p):
        return self.profit_total(p, self.unit_cost, self.q_0, self.k, self.a_q, self.b_q)
    
    
