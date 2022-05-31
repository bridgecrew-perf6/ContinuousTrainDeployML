
# signal = sim.signals_case_generation(n_transition_steps=500)
# print(f'Shape of signal {signal}')
import requests
import sys
from numpy import frombuffer


url = 'http://localhost:5002/timesteps'
initial_step_param = {'initial_step': 0}
r = requests.get(url=url , params=initial_step_param)
data = frombuffer(r.content)
print(data.shape)

# sys.exit('done for now')

initial_step_param = {'initial_step': 0}

url = 'http://localhost:5001/train'
r = requests.get(url=url , params=initial_step_param)
print(type(r))
print('debug')  