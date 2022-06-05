
# signal = sim.signals_case_generation(n_transition_steps=500)
# print(f'Shape of signal {signal}')
import json
import requests
import sys
from numpy import frombuffer, vstack, hstack
from json import loads as json_loads
from json import load as json_load 
import utils.preprocessing as prep
import utils.simulation_utils as sim
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam


# sys.exit('done for now')
# url = 'http://localhost:5001/train'
url = 'http://34.150.196.128:5001/train'
DEPLOY_PATIENCE = 4
patience_status = 0

for iteration, start_step in enumerate(range(0, 2600, 400)):

  initial_step_param = {'initial_step': start_step}  
  response = requests.get(url=url , params=initial_step_param)
  results = json_loads(response.json()) if response and response.status_code == 200 else None
  # results = json_loads(response.text)
  rmse_candidate, rmse_prod = results.values()  
  print(f'Iteration {iteration+1}:  {rmse_candidate}::{rmse_prod} @step: {start_step}')
  if rmse_candidate < rmse_prod:
    patience_status +=1
  else:
    patience_status = 0

  if patience_status >= DEPLOY_PATIENCE:
    print('DEPLOYMENT')  

print('done')