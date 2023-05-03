""" UPDATE LOGs
2023/04/24: Version 1.0
    - Enable training with BCELoss, treating racing prediction as 
      the binary classification problem, with output scoring as
      the ranking entity for each data.

2023/04/26: Version 1.1
    1. Enable training with no numeric features, i.e. pure collaborative with
       jockey, horse, trainer.

    2. Pairwise training 
        See another file

TODO:
1. SAVE MODEL
2. MORE TRAINING STRTEGIES
3. PARAM TUNING
4. Embedding Pretraining?

"""


""" IDEAS
task -> way: default as prediction
loss
model
lr
wd
"""

import os
import time
import argparse

import numpy as np
from torch import autograd, device, optim \
                  , FloatTensor, LongTensor

from data.load_data import DataSet
from model import BCELoss, MSELoss, LogSigmoidLoss
from model import LinEmbConcat, LinEmbDotProd, LinEmbElemProd, EmbMLP
from process import racing_champ, AveragePrecision

########################################################
###################### parse args ######################
########################################################

parser = argparse.ArgumentParser()
# Basics
parser.add_argument('--dataset_path', default='./horse/data/perform_full_feature.csv', type=str, help='Path of the dataset')
parser.add_argument('--do_categorization', default=True, type=bool, help='Categorize data into dummy/id')
parser.add_argument('--do_scale', default=True, type=bool, help='If use scaling to process some fields of the data')
parser.add_argument('--train_size', default=0.8, type=float, help='Perc of days of data to be trained')
parser.add_argument('--val_size', default=0.1, type=float, help='Perc of days of data to be treated as validation dataset')
parser.add_argument('--test_size', default=0.1, type=float, help='Perc of days of data to be tested')

# Model Offset
parser.add_argument('--model_name', default='EmbMLP', type=str, help='Name of model to be recalled')
parser.add_argument('--k_dim_field', default=4, type=int, help='Embedding dimension of field, dr')
parser.add_argument('--k_dim_id', default=32, type=int, help='Embedding dimension of horse, jockey, trainer')
parser.add_argument('--num_layers', default=2, type=int, help='Number of layers for MLP model')
parser.add_argument('--p_dropout', default=0.1, type=float, help='Probability of neurons to be turned of')
parser.add_argument('--layer_size_reduction', default=0.5, type=float, help='Layer size reduction for MLP')

# Compare Effect of Features
parser.add_argument('--use_numeric', default=False, type=bool, help='If use Numeric Features')
parser.add_argument('--use_best_feats', default=False, type=bool, help='If use the best subset of features')

# Training
parser.add_argument('--use_cuda', default=True, type=bool, help='If train on cuda')
parser.add_argument('--epoch', default=10, type=int, help='Num of epochs')
parser.add_argument('--batch_size', default=20, type=int, help='Size of each batch to be learnt')
parser.add_argument('--learning_rate', default=1e-5, type=float, help='The extent of learning rate in gradient descent')
parser.add_argument('--weight_decay', default=1e-3, type=float, help='Intensity of l2 regularization')
parser.add_argument('--out_src', default='./output', type=str, help='Where the model to be saved')

args = parser.parse_args()
print(f'ARGUMENTS: {args}\n')

########################################################
##################### Load Dataset #####################
########################################################
dataset = DataSet(args.dataset_path, scaling=args.do_scale, do_categorization=args.do_categorization
                  , use_best_feats=args.use_best_feats)
perc = [args.train_size, args.val_size, args.test_size]
train, val, test = dataset.my_train_val_test_split(perc)
n_dr = dataset.data['dr'].nunique()
n_field = dataset.data['field_going'].nunique()
n_jockey = dataset.data['jockey'].nunique()
n_horse = dataset.data['horse'].nunique()
n_trainer = dataset.data['trainer'].nunique()


########################################################
################### Global Variables ###################
########################################################
## Field Map
# TODO: More targets?
y_col = ['is_champ']
candidate_target_cols = ['is_champ', 'pla', 'speed', 'win_odds']
categ_cols = ['dr_ix', 'field_going', 'horse', 'jockey', 'trainer']
key_cols = ['race_key', 'race_date', 'dr']
# TODO: Adding more features!
if args.use_numeric:
    numerical_cols = [col for col in dataset.data.columns if col not in candidate_target_cols+categ_cols+key_cols]
else:
    numerical_cols = []
n_num_feats = len(numerical_cols)
# print(f'{n_num_feats} numeric cols: {numerical_cols}')
x_cols = numerical_cols + categ_cols
# print(f'All features: {x_cols}\n')

## Model Map
model_map = {
    'LinEmbConcat': LinEmbConcat
    , 'LinEmbDotProd': LinEmbDotProd
    , 'LinEmbElemProd': LinEmbElemProd
    , 'EmbMLP': EmbMLP
}
# parse model params
k_dim_field = args.k_dim_field
k_dim_id = args.k_dim_id
num_layers = args.num_layers
p_dropout = args.p_dropout

def merge_param(param1:dict, param2:dict):
    """ Merge 2 set of params """
    param = param1.copy()
    for key in param2:
        param[key] = param2[key]
    return param

base_params = { # public params
    'n_dr':n_dr
    , 'n_field':n_field
    , 'n_jockey':n_jockey
    , 'n_horse':n_horse
    , 'n_trainer':n_trainer
    , 'n_num_feats':n_num_feats
    , 'k_dim_field':k_dim_field
    , 'k_dim_id':k_dim_id
    , 'need_activation':True
}

model_param_dict = {
    'LinEmbConcat': base_params
    , 'LinEmbDotProd': base_params
    , 'LinEmbElemProd': base_params
    , 'EmbMLP': merge_param(base_params, {'num_layers':num_layers, 'p_dropout':p_dropout}) # specific params
}
model = model_map[args.model_name]
model_param = model_param_dict[args.model_name]
print(f'Model param: {model_param}')
model = model(**model_param) # pass dict of params to the model
print(f'MODEL: {model}\n')


########################################################
#################### Important Func ####################
########################################################

def get_feats(data, numerical_cols, y_col, use_cuda=False):
    compute_device = device('cuda') if use_cuda else device('cpu')

    x = FloatTensor(data[numerical_cols].values).to(compute_device)
    d = LongTensor(data['dr_ix'].values).to(compute_device)
    f = LongTensor(data['field_going'].values).to(compute_device)
    j = LongTensor(data['jockey'].values).to(compute_device)
    h = LongTensor(data['horse'].values).to(compute_device)
    t = LongTensor(data['trainer'].values).to(compute_device)
    y = FloatTensor(data[y_col].values).to(compute_device)

    X = (x, d, f, j, h, t)
    return X, y

def horse_data_loader(X, y, batch_size, shuffle):
    from torch.utils.data import DataLoader

    return DataLoader(list(zip(*X, y)), batch_size=batch_size, shuffle=shuffle)

def prep_eval_data(perform, use_cuda):
    """
    ((race_key, dr), (X, d, f, j, h, t))
    """ 
    keys = (perform[['race_key', 'dr']])
    X, y = get_feats(perform, numerical_cols, y_col, use_cuda)

    return keys, X

def computeAP(dataset, model, way='min', use_cuda=False):
    func = min if way=='min' else max
    ground_truth = racing_champ(dataset)
    result, X = prep_eval_data(dataset, use_cuda)
    score = model(*X)
    result['win'] = score.to(device('cpu')).detach().numpy()
    result = result.groupby(['race_key']) \
                .apply(lambda x: x[x['win']==func(x['win'])]) \
                .reset_index(drop=True)[['race_key', 'dr']]

    return AveragePrecision(result, ground_truth)


########################################################
######################### Preps ########################
########################################################
# compute device
use_cuda = args.use_cuda
compute_device = device('cuda') if use_cuda else device('cpu')
# data
X_train, y_train = get_feats(train, numerical_cols, y_col, use_cuda)
# model, optim
model = model.to(compute_device)
opt = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
# loss
train_loss_by_ep = []
val_ap_by_ep = []
test_ap_by_ep = []

for ep in range(args.epoch):
    t0 = time.time()
    ep_loss = []
    for batch_data in horse_data_loader(X_train, y_train, args.batch_size, shuffle=True):
        x, d, f, j, h, t, y = batch_data
        model.zero_grad()

        y_pred = model(x, d, f, j, h, t)
        loss = BCELoss(y_pred, y)
        
        opt.zero_grad()
        loss.backward()
        opt.step()

        ep_loss.append(loss.data.to(device('cpu')).tolist())
#         print(ep_loss)

    train_loss_by_ep.append(np.mean(ep_loss))
    
    # compute AP
    val_ap_by_ep.append(computeAP(val, model, way='max', use_cuda=use_cuda))
    test_ap_by_ep.append(computeAP(test, model, way='max', use_cuda=use_cuda))
    
    t1 = time.time()
    print(f'[{round(t1-t0, 3)}s] Iter={ep}, train loss={round(train_loss_by_ep[-1], 3)}')
    print(f'\t [VAL] AP={round(val_ap_by_ep[-1], 3)}; [TEST] AP={round(test_ap_by_ep[-1], 3)}')