import os, argparse
from itertools import product
import pandas as pd
import numpy as np

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
parser.add_argument('--train_file_path', default='./horse/ml_train.py')
args = parser.parse_args()

common_grid = {'learning_rate':log_lin_space(-5, 0, 6, 1), 'weight_decay':log_lin_space(-5, 0, 6, 1)}

models = {
    'LinEmbConcat': common_grid.copy()
    , 'LinEmbDotProd': common_grid.copy()
    , 'LinEmbElemProd': common_grid.copy()
    , 'EmbMLP': common_grid.copy()
}

def range_to_list(range_list):
    return [x for x in np.arange(range_list[0], range_list[1], range_list[-1])]

def get_result(model:str):
    print('Training {}....'.format(model))
    for model_name in models.keys():
        if model_name == model:
            result = {'AP':[]}
            temp = []
            for params_name in models[model_name].keys():
                result[params_name] = []
                if isinstance(models[model_name][params_name][0], str):
                    temp.append(models[model_name][params_name])
                else:
                    temp.append(range_to_list(models[model_name][params_name]))
            combinations = product(*temp)
            keys = [key for key in models[model_name].keys()]
            for combine in combinations:
                cmd = 'python {} --model {}'.format(args.train_file_path, model_name)
                count = 0
                temp = {}
                for key in keys:
                    temp[key] = []
                for value in combine:
                    cmd += ' --{paramname} {paramvalue}'.format(paramname=keys[count],paramvalue=value)
                    result[keys[count]].append(value)
                    count += 1
                print(cmd)
                cmd_result = os.popen(cmd)
                #print(cmd_result.readlines())
                ap = cmd_result.readlines()[1].split(':')[-1].strip()
                result['AP'].append(ap)
            print(result)
        else:
            continue
    pd.DataFrame.from_dict(result).to_csv('{}{}.csv'.format(args.result_path, model))


get_result(args.model_name)
