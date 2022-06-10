from typing import Union, Tuple
from numpy import linspace, concatenate, pi, cos, ndarray, array, sin, random, cumsum

def generate_raw_signal(max_degrees: int, steps_ratio: Union[int, float]) -> Tuple:

  steps = int(max_degrees/steps_ratio) 
  # print(f'max_degrees: {max_degrees} steps: {steps}')
  degrees = linspace(0, max_degrees, steps, endpoint=False).astype(int)
  reversed_degrees = linspace(max_degrees, 0, steps, endpoint=False).astype(int)  

  return degrees, reversed_degrees


def to_radian(degrees):
  return degrees * (pi / 180)


def stack_data(degree_data: Union[array, Tuple], n_iters: int, to_radians: bool = False) -> array:
  
    if isinstance(degree_data, ndarray):
        data = degree_data
        for i in range(n_iters):
            data = concatenate((data, degree_data))

    else: 
        data = degree_data[0]
        for i in range(n_iters):
            if i % 2 == 0:
                data = concatenate((data, degree_data[1]))
            else:
                data = concatenate((data, degree_data[0]))

    if to_radians:
        data = to_radian(data)

    return data


def cosine(data: array) -> array:
    return cos(data)


def sine(data: array) -> array:
    return sin(data)


def data_generator(data: array, n_timesteps: int) -> array:
    for i in range(data.shape[0]):
        yield data[i]


def timesteps_generator(data: array, start_timestep: int, n_timesteps: int = 5000):
        return data[start_timestep:start_timestep + n_timesteps]


def skewed_noise(n_samples: int, params: tuple=(1,1), dist_type='gamma', random_state=43):    
  random.seed(random_state)

  if dist_type == "normal":
    mu, sigma = params
    random_signal = random.normal(mu, sigma, n_samples)    

  elif dist_type == "gamma":
    shape, scale = params 
    random_signal = random.gamma(shape, scale, n_samples)

  return random_signal


def signal_fusion(cyclic_signal: array, 
                  random_signal: array, 
                  size_coef: float = 1, 
                  starting_point: Union[float, None] = None, 
                  cumulative: bool = True) -> array:

  s_mixed = cyclic_signal * random_signal * size_coef
  
  if starting_point is not None:
    # first_point = s_mixed[0]
    s_mixed[0] = starting_point
    # print(f'{first_point} - {starting_point} - {s_mixed[0]}')

  if cumulative:
    s_mixed = cumsum(s_mixed)
  
  return s_mixed


def signals_case_generation(n_transition_steps: int = 500, verbose: bool = False):
    """
    Returns the exact same data points used for the raid prototyping
    n_transition_steps: determines the timesteps between the full end of the
        signal one and the full start of signal 2. 
    """
    raw_1 = generate_raw_signal(max_degrees=360, steps_ratio=0.02)
    data_1 = stack_data(raw_1, 10, to_radians=False)
    cos_signal = cosine(data_1)
    random_signal_1 = skewed_noise(len(cos_signal), params=(1.1, .4))
    random_signal_1 = 1- random_signal_1
    s_mixed = signal_fusion(cos_signal, random_signal_1, size_coef=.7)

    if verbose:
        print(f'cos_signal shape: {cos_signal.shape[0]}')
        print(cos_signal[:5])

    raw_2 = generate_raw_signal(max_degrees=360, steps_ratio=0.07)
    data_2 = stack_data(raw_2, 2, to_radians=False)
    sin_signal =sine(data_2)    
    random_signal_2 = skewed_noise(len(sin_signal), params=(.7, .7))    
    random_signal_2 = 1- random_signal_2
    s_production = signal_fusion(sin_signal, random_signal_2, size_coef=1.2, starting_point=s_mixed[-1])

    coef = linspace(0, 1, n_transition_steps).astype(float)
    transition_signal = (s_mixed[-n_transition_steps:]*(1-coef) + s_production[:n_transition_steps]*coef)
    steps_2nd_signal = 5000
    ix = s_mixed.shape[0] - 5500
    ix_final = s_mixed.shape[0] - n_transition_steps
    concat_signal = concatenate((s_mixed[ix:ix_final], transition_signal, s_production[n_transition_steps:n_transition_steps+steps_2nd_signal]))

    print(f's_mixed: {s_mixed[ix:ix_final].shape} | transition_signal: {transition_signal.shape} | s_production: {s_production[n_transition_steps:n_transition_steps+steps_2nd_signal].shape} | concat: {concat_signal.shape}')

    transition_signal_2 = (s_production[steps_2nd_signal+n_transition_steps:steps_2nd_signal+n_transition_steps*2]*(1-coef) + s_mixed[ix:ix+n_transition_steps]*coef)

    concat_signal_full = concatenate((concat_signal, transition_signal_2, s_mixed[ix+n_transition_steps:]))

    print(f'transition_2: {transition_signal_2.shape} | s_mixed: {s_mixed[ix+n_transition_steps:].shape}')

    # if verbose: 
    print(f'Signal 1 shape: {concat_signal.shape}')
    print(f'Signal full shape: {concat_signal_full.shape}')

    return concat_signal_full