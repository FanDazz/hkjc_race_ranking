import argparse
import pandas as pd
# from horse.process import train_test_split
from model import HKJC_models
from data.load_data import DataSet, train_val_test_split


######################################################
#               Selection Parameters                 #
######################################################
parser = argparse.ArgumentParser(description='HKJC')
# dataset
parser.add_argument('--dataset_path', default='./horse/data/perform_full_feature.csv', type=str, help='Path of the dataset')
parser.add_argument('--do_scale', default=True, type=bool, help='If use scaling to process some fields of the data')
parser.add_argument('--do_categorization', default=True, type=bool, help='Categorize data into dummy/id')
parser.add_argument('--use_best_feats', default=True, type=bool, help='If use the best subset of features')

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
parser.add_argument('--penalty', type=str, default="l1")
parser.add_argument('--solver', type=str, default="liblinear")
parser.add_argument('--C', type=float, default=0.5)
parser.add_argument('--max_iter', type=int, default=1000)
parser.add_argument('--criterion', type=str, default="entropy")
parser.add_argument('--random_state', type=int, default=8017)
parser.add_argument('--splitter', type=str, default='random')
parser.add_argument('--max_depth', type=int, default=10)
parser.add_argument('--min_samples_leaf', type=int, default=10)
parser.add_argument('--min_samples_split', type=int, default=10)
parser.add_argument('--n_estimators', type=int, default=50)
parser.add_argument('--learning_rate', type=float, default=0.1)
parser.add_argument('--algorithm', type=str, default='SAMME')
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--loss', type=str, default='linear')
parser.add_argument('--tree_method', type=str, default="gpu_hist")
parser.add_argument('--min_child_weight', type=int, default=1)
parser.add_argument('--subsample', type=float, default=0.8)
parser.add_argument('--colsample_bytree', type=float, default=0.8)
parser.add_argument('--reg_alpha', type=float, default=0.0001)
parser.add_argument('--reg_lambda', type=float, default=0.0001)
parser.add_argument('--objective', type=str, default="binary:logistic")
parser.add_argument('--eval_metric', type=str, default='mae')
args = parser.parse_args()
print(args)

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
y_cols = ['is_champ', 'pla', 'finish_time', 'speed']
date_cols = ['race_key', 'race_date']
get_x_cols = lambda x: [col for col in x if (col not in y_cols) and (col not in date_cols)]

dataset = DataSet(args.dataset_path, scaling=args.do_scale, do_categorization=args.do_categorization
                  , use_best_feats=args.use_best_feats)
dm_perform_train, dm_perform_val, dm_perform_test = train_val_test_split(dataset.data, 'race_date'
                                                                          , perc=[args.train_size, args.val_size, args.test_size])
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