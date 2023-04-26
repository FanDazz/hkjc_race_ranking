import argparse
import pandas as pd
# from horse.process import train_test_split
from model import HKJC_models


######################################################
#               Selection Parameters                 #
######################################################
parser = argparse.ArgumentParser(description='HKJC')
parser.add_argument('--data_path',default='./horse/data/perform_clean.csv', help='Data path')
parser.add_argument('--if_odds', default=False, help='The data will contain odds')
parser.add_argument('--target', default="is_champ", help='Targets: is_champ, finish_time, speed')
# parser.add_argument('--split_rate',default=0.1, help='Rate for spliting test set.')
parser.add_argument('--train_size', default=0.8, type=float, help='Perc of days of data to be trained')
parser.add_argument('--val_size', default=0.1, type=float, help='Perc of days of data to be treated as validation dataset')
parser.add_argument('--test_size', default=0.1, type=float, help='Perc of days of data to be tested')

parser.add_argument('--model',default="logistic",help='selection of models')
######################################################
#                  Models Parameters                 #
######################################################
parser.add_argument('--penalty',default="l1")
parser.add_argument('--solver',default="liblinear")
parser.add_argument('--C',default=0.5)
parser.add_argument('--max_iter',default=1000)
parser.add_argument('--criterion',default="entropy")
parser.add_argument('--random_state',default=8017)
parser.add_argument('--splitter',default='random')
parser.add_argument('--max_depth',default=10)
parser.add_argument('--min_samples_leaf',default=10)
parser.add_argument('--min_samples_split',default=10)
parser.add_argument('--n_estimators',default=50)
parser.add_argument('--learning_rate',default=0.1)
parser.add_argument('--algorithm',default='SAMME.R')
parser.add_argument('--base_estimator',default='deprecated')
parser.add_argument('--alpha',default=1.0)
parser.add_argument('--loss',default='linear')
parser.add_argument('--tree_method',default="gpu_hist")
parser.add_argument('--min_child_weight',default=1)
parser.add_argument('--subsample',default=0.8)
parser.add_argument('--colsample_bytree',default=0.8)
parser.add_argument('--reg_alpha',default=0.0001)
parser.add_argument('--reg_lambda',default=0.0001)
parser.add_argument('--objective',default="binary:logistic")
parser.add_argument('--eval_metric',default='mae')
args = parser.parse_args()

param = {
    "penalty": args.penalty,
    "solver": args.solver,
    "C": args.C,
    "max_iter": args.max_iter,
    "criterion": args.criterion,
    "random_state": args.random_state,
    "splitter": args.splitter,
    "max_depth": args.max_depth,
    "min_samples_leaf": args.min_samples_leaf,
    "min_samples_split": args.min_samples_split,
    "n_estimators": args.n_estimators,
    "learning_rate": args.learning_rate,
    "algorithm": args.algorithm,
    "base_estimator": args.base_estimator,
    "alpha": args.alpha,
    "loss": args.loss,
    "tree_method": args.tree_method,
    "min_child_weight": args.min_child_weight,
    "subsample": args.subsample,
    "colsample_bytree": args.colsample_bytree,
    "reg_alpha": args.reg_alpha,
    "reg_lambda": args.reg_lambda,
    "objective": args.objective,
    "eval_metric": args.eval_metric
}
######################################################
#             Generate performing data               #
######################################################
def _train_test_split(data, col, perc=[0.8, 0.1, 0.1]):
    time_map = data[col].drop_duplicates()
    n_dates = time_map.__len__()
    cut_point1 = int(perc[0]*n_dates)
    cut_point2 = int((perc[0]+perc[1])*n_dates)

    train_date, val_date, test_date = time_map[:cut_point1], time_map[cut_point1:cut_point2], time_map[cut_point2:]
    train, val, test = data.merge(train_date, how='inner'), data.merge(val_date, how='inner'), data.merge(test_date, how='inner')
    
    return train, val, test

perform = pd.read_csv(args.data_path, sep=',', encoding='utf-8')
perform['speed'] = perform['distance'] / perform['finish_time']
perform['is_champ'] = perform['pla'].apply(lambda x: 1 if x == 1 else 0)
y_cols = ['is_champ', 'pla', 'finish_time', 'speed']
date_cols = ['race_key', 'race_date']
get_x_cols = lambda x: [col for col in x if (col not in y_cols) and (col not in date_cols)]

dm_perform = pd.get_dummies(perform, columns=['field_going', 'course_type', 'horse', 'jockey', 'trainer'])
# dm_perform_train, dm_perform_test = train_test_split(dm_perform, 'race_date', args.split_rate)
# dm_perform_train, dm_perform_val = train_test_split(dm_perform_train, 'race_date', args.split_rate)
dm_perform_train, dm_perform_val, dm_perform_test = _train_test_split(dm_perform, 'race_date', [args.train_size, args.val_size, args.test_size])

remove_odds = lambda x: [col for col in x if col != 'win_odds']



######################################################
#                  Data Odds Process                 #
######################################################
x_cols = get_x_cols(dm_perform_train.columns)
if args.if_odds == True:
    X, y = dm_perform_train[x_cols], dm_perform_train[args.target]
else:
    x_cols = remove_odds(x_cols)
    X, y = dm_perform_train[x_cols], dm_perform_train[args.target]


######################################################
#                  Classifier Models                 #
######################################################
models = HKJC_models(param)
models.cal_result(args.model, args.target, X, y, dm_perform_val, x_cols)