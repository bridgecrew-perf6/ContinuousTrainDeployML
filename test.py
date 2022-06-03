
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
url = 'http://localhost:5001/train'
DEPLOY_PATIENCE = 4
patience_status = 0

for iteration, start_step in enumerate(range(0, 7600, 400)):

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

sys.exit('Done')
data = sim.signals_case_generation(n_transition_steps=500)
sequenced = prep.rolling_window(data, 100)
X, y = prep.seq2inputs(sequenced)
X_train, X_val, X_test, y_train, y_val, y_test = prep.split_dataset(X, y, split_size=100, verbose=True)
X_train = vstack((X_train, X_val))
y_train = hstack((y_train, y_val))

candidate = load_model('src/models/model_one')
candidate.compile(
  Adam(learning_rate=0.0001),
  loss= MeanSquaredError(),  
  metrics=['mse']  
)

candidate.fit(X_train, y_train, epochs=10)

rmse = round(candidate.evaluate(X_test, y_test, verbose=0)[0],2)




print("debug")
