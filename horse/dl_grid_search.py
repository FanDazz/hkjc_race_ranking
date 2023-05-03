import os, argparse
from itertools import product
import pandas as pd
import numpy as np
import re
import time

def log_lin_space(log_start, log_end, log_N, lin_N):
    """ Generate Tuning Space for params
        : param log_start: int
            the begin power for base 10
        : param log_end: int
            the end power for base 10
        : param log_N: int
            numbers to be generated with same power from [10^log_start, 10^log_end]
        : param lin_N: int
            numbers to be generated with same steps and same magnitude, within the logspace

        : return: list
            list of params generated
        ---

        If we with to generate 1 number between [10^-5, 10^-4], and for each
        number generated, find the following 5 numbers with same magnitude, call:

        >>> log_lin_space(-5, -4, 1, 5) 
            array([1.e-05, 2.e-05, 3.e-05, 4.e-05, 5.e-05]) 

        If we with to generate 3 number between [10^-5, 10^-2], and for each
        number generated, find the following 3 numbers with same magnitude, call:

        >>> log_lin_space(-5, -2, 3, 3) 
            array([1.00000000e-05, 2.00000000e-05, 3.00000000e-05, 3.16227766e-04,
                   6.32455532e-04, 9.48683298e-04, 1.00000000e-02, 2.00000000e-02,
                   3.00000000e-02])
    """
    log_space = np.logspace(log_start, log_end, log_N, base=10)
    lin_space = []
    for elem in log_space:
        lin_end = lin_N*elem
        lin_space.append(np.linspace(elem, lin_end, lin_N))

    return np.concatenate(lin_space)

parser = argparse.ArgumentParser(description='GridSearch')
parser.add_argument('--model_name')
parser.add_argument('--result_path', default='')
parser.add_argument('--train_file_path', default='./horse/train.py')
args = parser.parse_args()

common_grid = {
    # fixed params
    'k_dim_field':[4], 'k_dim_id':[32], 'batch_size':[20], 'epoch':[50]
    # tuned params
    , 'use_numeric': [False, True]
    , 'use_best_feats': [False, True]
    , 'learning_rate':log_lin_space(-4, 0, 5, 1), 'weight_decay':log_lin_space(-4, 0, 5, 1)}

model_param_dict = {
    'LinEmbConcat': common_grid.copy()
    , 'LinEmbDotProd': common_grid.copy()
    , 'LinEmbElemProd': common_grid.copy()
    , 'EmbMLP': common_grid.copy()
}

def range_to_list(range_list):
    return [x for x in np.arange(range_list[0], range_list[1], range_list[-1])]

def get_result(model:str):
    print('Training {}....'.format(model))
    params_to_tune = model_param_dict[model]
    param_names = list(params_to_tune.keys()) 
    param_comb_generator = product(*params_to_tune.values())
    for param_comb in param_comb_generator:
        _param_dict = dict(zip(param_names, param_comb))
        if _param_dict['use_numeric']==False and _param_dict['use_best_feats']==True:
            continue
        
        t0=time.time()
        # result saving dict for each model
        result = {'ep':[], 'val_AP':[], 'test_AP':[]}
        # generate command
        cmd = f'python {args.train_file_path} --model {model}'
        for param in _param_dict:
            param_value = _param_dict[param]
            cmd += f' --{param} {param_value}'
        print(cmd)
        cmd_result = os.popen(cmd)
#         print(cmd_result)
        for line in cmd_result.readlines():
            try:
                ep_ = re.findall(r'Iter=(.*?),', line)[0]
                val_AP_ = re.findall(r'\[VAL\] AP=(.*?);', line)[0]
                test_AP_ = re.findall(r'\[TEST\] AP=(.*?);', line)[0]
            
                result['ep'].append(int(ep_))
                result['val_AP'].append(float(val_AP_))
                result['test_AP'].append(float(test_AP_))
            except:
                continue
        
        # file name
        
        file_name = f'{model}'
        for ix, value in enumerate(param_comb):
            file_name += f'_{param_names[ix]}_{value}'
        file_name += '.csv'
        
        pd.DataFrame.from_dict(result).to_csv(f'{args.result_path}/{file_name}', index=False)
        t1 = time.time()
        print(f'DONE in {round(t1-t0, 2)}s.')

if __name__ == "__main__":
    # 1) make dir
    path = args.result_path
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
        
    # 2) grid search
    get_result(args.model_name)
