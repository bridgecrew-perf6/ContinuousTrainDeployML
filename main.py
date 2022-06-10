
import requests
import utils.metadata as meta
from time import sleep
from json import loads as json_loads


server = 'http://localhost'
# server = 'http://34.150.196.128'

trainer_url = f'{server}:5001/' 
train_url = trainer_url + 'train'
predict_url = trainer_url + 'predict'
deploy_url = trainer_url + 'deploy-candidate'
production_url = f'{server}:5000/'

train_length = 4000
evaluate_length = 100
n_rolling_window = 100

iteration_step = 50

DEPLOY_THRESHOLD = 2/3
DEPLOY_PATIENCE = 3


def main():
  patience_status = 0
  for start_step in range(0, 10000, iteration_step):

    prod_health = requests.get(production_url+'health')
    if prod_health.status_code != 200:
      print('Production server down')
      sleep(10)
      continue

    step_param = {'initial_step': start_step, 'n_timesteps': train_length + evaluate_length + n_rolling_window}  
    response = requests.get(url=train_url , params=step_param)
    results = response.json() if response and response.status_code == 200 else None

    pred_param = {'initial_step': start_step, 'n_timesteps': iteration_step+n_rolling_window}  
    r = requests.get(url=predict_url, params=pred_param)

    rmse_candidate, rmse_prod = results.values()  
    rmse_proportion = round((rmse_prod-rmse_candidate)/rmse_prod, 2)
    print(f'train-step: {start_step} to {start_step+train_length}  evaluate: {start_step+train_length+1} to {start_step+ train_length+evaluate_length} RESULTS: {rmse_candidate} :: {rmse_prod} proportion: {rmse_proportion}')

    if rmse_prod > 1 and rmse_proportion > DEPLOY_THRESHOLD:
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