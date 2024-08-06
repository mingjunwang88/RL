import pickle 
import numpy as np
import matplotlib.pyplot as plt

ppo = []
for i in range(25):
    with open('output/mappo/' + f'mappo_{i}', 'rb') as f: 
        mylist = pickle.load(f) 
        ppo.append(mylist)
ppo_array = np.array(ppo).mean(axis=0)
print(ppo_array.shape)

ppo_indep = []
for i in range(25):
    with open('output/mappo/' + f'mappo_indep_{i}', 'rb') as f: 
        mylist = pickle.load(f) 
        ppo_indep.append(mylist)
ppo_indep_array = np.array(ppo_indep).mean(axis=0)
print(ppo_indep_array.shape)

maac = []
for i in range(25):
    with open('output/maac/' + f'maac_{i}', 'rb') as f: 
        mylist = pickle.load(f) 
        maac.append(mylist)
maac_array = np.array(maac).mean(axis=0)
print(maac_array.shape)

maac_indep = []
for i in range(25):
    with open('output/maac/' + f'maac_indep_{i}', 'rb') as f: 
        mylist = pickle.load(f) 
        maac_indep.append(mylist)
maac_array_indep = np.array(maac_indep).mean(axis=0)
print(maac_array_indep.shape)

final = np.vstack([ppo_array, ppo_indep_array])
print(final.shape)
print(final)

plt.plot(ppo_array, label='Mappo_Centralized')
plt.plot(ppo_indep_array, label='Mappo_Indepdent')
plt.plot(maac_array, label='Maac_Cantralized')
plt.plot(maac_array_indep, label='Maac_Indepdent')
plt.legend()
plt.show()
