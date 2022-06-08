
import requests
from json import loads as json_loads
import utils.metadata as meta
from time import sleep

server = 'http://localhost'
# server = 'http://34.150.196.128'

trainer_url = f'{server}:5001/' 
train_url = trainer_url + 'train'
deploy_url = trainer_url + 'deploy-candidate'
production_url = f'{server}:5000/'

train_length = 4000
evaluate_length = 100

DEPLOY_THRESHOLD = round(2/3,2)
DEPLOY_PATIENCE = 3

patience_status = 0

def main():
  for start_step in range(0, 10000, 50):

    prod_health = requests.get(production_url+'health')
    if prod_health.status_code != 200:
      print('Production server down')
      sleep(10)
      continue

    initial_step_param = {'initial_step': start_step}  
    response = requests.get(url=train_url , params=initial_step_param)
    results = json_loads(response.json()) if response and response.status_code == 200 else None

    rmse_candidate, rmse_prod = results.values()  
    rmse_proportion = round((rmse_prod-rmse_candidate)/rmse_prod, 2)
    print(f'train-step: {start_step} to {start_step+train_length}  evaluate: {start_step+train_length+1} to {start_step+ train_length+evaluate_length} RESULTS: {rmse_candidate} :: {rmse_prod} proportion: {rmse_proportion}')

    if rmse_proportion > DEPLOY_THRESHOLD:
      patience_status +=1
    else:
      patience_status = 0

    if patience_status >= DEPLOY_PATIENCE:
      print('DEPLOYMENT')
      response = requests.get(url=deploy_url)
      meta.restart_container(name='production')
      sleep(10)  

if __name__ == "__main__":

  main()
  print('done')