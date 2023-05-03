import os, argparse
from itertools import product
import pandas as pd
import numpy as np
import re

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

common_grid = {'learning_rate':log_lin_space(-5, 0, 6, 1), 'weight_decay':log_lin_space(-5, 0, 6, 1)}

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
        # result saving dict for each model
        result = {'ep':[], 'val_AP':[], 'test_AP':[]}
        # generate command
        cmd = f'python {args.train_file_path} --model {model}'
        for ix, value in enumerate(param_comb):
            cmd += f' --{param_names[ix]} {value}'
        print(cmd)
        cmd_result = os.popen(cmd)
        for line in cmd_result.readlines():
            ep_ = re.findall(r'Iter=(.*?),', line)
            val_AP_ = re.findall(r'\[VAL\] AP=(.*?);', line)
            test_AP_ = re.findall(r'\[TEST\] AP=(.*?);', line)
            
            result['ep'].append(ep_)
            result['val_AP'].append(val_AP_)
            result['test_AP'].append(test_AP_)
        
        # file name
        
        file_name = f'{model}'
        for ix, value in enumerate(param_comb):
            file_name += f'_{param_names[ix]}_{value}'
        file_name += '.csv'
        
        pd.DataFrame.from_dict(result).to_csv(f'{args.result_path}/{file_name}', index=False)


if __name__ == "__main__":
    # 1) make dir, if not exist
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)
        
    # 2) grid search
    get_result(args.model_name)
