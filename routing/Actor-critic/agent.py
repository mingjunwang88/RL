import time
import torch as t 

from ppo import PPO
import importlib
import env_one_hot
importlib.reload(env_one_hot)
from env_one_hot import CitizenBankEnv
start_time = time.time()
import gymnasium as gym
from copy import copy

env = CitizenBankEnv(1)
#env_name = 'CartPole-v1'
#env = gym.make(env_name)

has_continuous_action_space = False

# state space dimension
state_dim = env.costs.shape[0]
# action space dimension
if has_continuous_action_space:
    action_dim = state_dim
else:
    action_dim = state_dim

lr_actor = 0.0003 #0.0003       # learning rate for actor network
lr_critic = 0.001       # learning rate for critic network
eps_clip = 0.2          # clip parameter for PPO
gamma = 0.99            # discount factor
K_epochs = 80 #80            # update policy for K epochs in one PPO update
steps_update = 4000
num_episodes = 100
steps_print = 100

class Agent:
    def __init__(self):
        self.lst = t.zeros(env.num_devices)

    def reset_memmory(self):
        self.memmory = []
        
    def remember_memmory(self, pos):
        self.memmory.append(pos)
        
    def available_actions(self, current_pos):
        actions = [i for i, j in enumerate(env.costs[current_pos,]) if j > 0]
        actions = [i for i in actions if i not in self.memmory]

        actions_ = copy(self.lst)
        actions_[actions] = 1
        return actions_.type(t.bool)
    
    def one_hot(self,pos):
        one_hot = copy(self.lst)
        one_hot[pos] = 1.
        return one_hot

    def train(self,):
        ppo = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6)
        
        time_step = 0
        num_rewards = []
        # training loop
        for i in range(num_episodes):
            total_reward = 0
            for state in range(env.num_devices):
                self.reset_memmory()
                done = False
                trun = False 
                #state, _ = env.reset()
                current_ep_reward = 0

                while not (done or trun):
                    self.remember_memmory(state)
                    available_actions = self.available_actions(state)
                    # select action with policy
                    state_one_hot = self.one_hot(state)
                    action = ppo.select_action(state_one_hot, available_actions)

                    state_next, reward, done, trun, _ = env.step(action)

                    # saving reward and is_terminals
                    ppo.buffer.rewards.append(reward)
                    ppo.buffer.is_terminals.append(done)
                    ppo.buffer.states_next.append(t.tensor(state_next, dtype=t.float))

                    time_step +=1
                    total_reward += reward

                    # update PPO agent
                    if time_step % steps_update == 0:
                        ppo.update(tdzero=False)

                    state = state_next

                    """
                    # if continuous action space; then decay action std of ouput action distribution
                    if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                        ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)
                    """
                #if i % steps_print == 0:
                #    print(f'episode: {i}: ', current_ep_reward)
            num_rewards.append(total_reward)

        x = [i+1 for i in range(num_episodes)]
        ppo.plot_learning_curve(x, num_rewards)


if __name__ == '__main__':
    agent = Agent()
    agent.train()
        

        