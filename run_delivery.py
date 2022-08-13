import importlib
import all
importlib.reload(all)
from all import *
from delivery import DeliveryEnvironment

import env
importlib.reload(env)
from env import DeliveryEnv


env = DeliveryEnv(100, 200, 'distance')
q_agent = QAgent(env.n_stops, env.n_stops)

agent = Agent(q_agent, env)

agent.train(1000)


agent.roll_out(19)
