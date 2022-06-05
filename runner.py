
import requests
from json import loads as json_loads




# sys.exit('done for now')
server_url = 'http://localhost:5001/' 
# url = 'http://34.150.196.128:5001/train'
train_url = server_url + 'train'
deploy_url = server_url + 'deploy-candidate'

DEPLOY_PATIENCE = 4
patience_status = 0

for iteration, start_step in enumerate(range(0, 7000, 400)):

  initial_step_param = {'initial_step': start_step}  
  response = requests.get(url=train_url , params=initial_step_param)
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
    response = requests.get(url=deploy_url, params=initial_step_param)

print('done')