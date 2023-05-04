""" UPDATE LOGs
2023/04/24: Version 1.0
    - Enable training with BCELoss, treating racing prediction as 
      the binary classification problem, with output scoring as
      the ranking entity for each data.

2023/04/26: Version 1.1
    1. Enable training with no numeric features, i.e. pure collaborative with
       jockey, horse, trainer.

    2. Pairwise training 

TODO:
1. SAVE MODEL
2. MORE TRAINING STRTEGIES
3. PARAM TUNING
4. Embedding Pretraining?

"""

import os
import time
import argparse

import numpy as np
import pandas as pd
import torch
from torch import autograd, device, optim \
                  , FloatTensor, LongTensor

from data.load_data import DataSet
from model import BCELoss, MSELoss, PairwiseLoss
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
# - pairwise training settings
parser.add_argument('--do_pairsise', default=True, type=bool, help='If use pairwise training')
parser.add_argument('--sampling_perc', default=0.8, type=float, help='Sampling percentage for pairwise learning')

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
# print(f'ARGUMENTS: {args}\n')

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
    , 'need_activation':False
}

model_param_dict = {
    'LinEmbConcat': base_params
    , 'LinEmbDotProd': base_params
    , 'LinEmbElemProd': base_params
    , 'EmbMLP': merge_param(base_params, {'num_layers':num_layers, 'p_dropout':p_dropout}) # specific params
}
model = model_map[args.model_name]
model_param = model_param_dict[args.model_name]
model = model(**model_param) # pass dict of params to the model
# print(f'MODEL: {model}\n')


########################################################
#################### Important Func ####################
########################################################
def get_feats(data, numerical_cols, y_col, use_cuda=False, call_eval=False):
    compute_device = device('cuda') if use_cuda else device('cpu')

    x = FloatTensor(data[numerical_cols].values)
    if not call_eval:
        d = LongTensor(data['dr_ix'].values.reshape([-1, 1]))
        f = LongTensor(data['field_going'].values.reshape([-1, 1]))
        j = LongTensor(data['jockey'].values.reshape([-1, 1]))
        h = LongTensor(data['horse'].values.reshape([-1, 1]))
        t = LongTensor(data['trainer'].values.reshape([-1, 1]))
    else:
#         print('evaling')
        d = LongTensor(data['dr_ix'].values)
        f = LongTensor(data['field_going'].values)
        j = LongTensor(data['jockey'].values)
        h = LongTensor(data['horse'].values)
        t = LongTensor(data['trainer'].values)
    
    y = FloatTensor(data[y_col].values).to(compute_device)

    X = (x.to(compute_device)
         , d.to(compute_device)
         , f.to(compute_device)
         , j.to(compute_device)
         , h.to(compute_device)
         , t.to(compute_device))
    return X, y

def horse_pairwise_data_loader(X1, X2, batch_size, shuffle):
    from torch.utils.data import DataLoader
    cat_X1 = torch.concat(X1, 1)
    cat_X2 = torch.concat(X2, 1)

    return DataLoader(list(zip(cat_X1, cat_X2)), batch_size=batch_size, shuffle=shuffle)

def split_cols(data):
    x, (d, f, j, h, t) = data[:, :-5], data[:, -5:].T
    d = d.long()
    f = f.long()
    j = j.long()
    h = h.long()
    t = t.long()
    
    return (x, d, f, j, h, t)

def prep_eval_data(perform, use_cuda, call_eval=False):
    """
    ((race_key, dr), (X, d, f, j, h, t))
    """ 
    keys = (perform[['race_key', 'dr']])
    X, y = get_feats(perform, numerical_cols, y_col, use_cuda, call_eval=call_eval)

    return keys, X

def computeAP(dataset, model, way='min', use_cuda=False):
    func = min if way=='min' else max
    ground_truth = racing_champ(dataset)
    result, X = prep_eval_data(dataset, use_cuda, call_eval=True)
    score = model(*X)
    result['win'] = score.to(device('cpu')).detach().numpy()
    result = result.groupby(['race_key']) \
                .apply(lambda x: x[x['win']==func(x['win'])]) \
                .reset_index(drop=True)[['race_key', 'dr']]

    return AveragePrecision(result, ground_truth)

def pairwise_sampler(data, sample_perc=0.5):
    from numpy.random import randint
    
    data_race_participants = data.groupby(['race_key'])['dr'].max()
    data_remix = data.drop_duplicates().set_index(['race_key', 'pla'])
#     print(f'data_remix shape: {data_remix.shape}')

    base_index = []
    sample_index = []

    race_keys = data_race_participants.index.tolist()
    for race_key in race_keys:
        num_of_participants = data_race_participants[race_key]
        for ix in range(1, num_of_participants+1):
            num_samples = int(sample_perc*(num_of_participants-ix))
            if num_samples<1:
                continue
            sample_ixs = randint(ix+1, num_of_participants+1, num_samples)
            base_index.extend([(race_key, ix) for _ in sample_ixs])
            sample_index.extend([(race_key, sample_ix) for sample_ix in sample_ixs])

    base_index = pd.Index(base_index)
    sample_index = pd.Index(sample_index)

    valid_ix_cond1 = base_index.isin(data_remix.index)
    valid_ix_cond2 = sample_index.isin(data_remix.index)
    valid_ix = (valid_ix_cond1) & (valid_ix_cond2) # <class 'numpy.ndarray'>
    
    base_valid_ix = base_index[valid_ix]
    sample_valid_ix = sample_index[valid_ix]
    
    sampled_x1 = data_remix.loc[base_valid_ix].reset_index()
    sampled_x2 = data_remix.loc[sample_valid_ix].reset_index()

    return sampled_x1, sampled_x2


########################################################
######################### Preps ########################
########################################################
# compute device
use_cuda = args.use_cuda
compute_device = device('cuda') if use_cuda else device('cpu')
# data
# model, optim
model = model.to(compute_device)
opt = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
# loss
train_loss_by_ep = []
val_ap_by_ep = []
test_ap_by_ep = []

train = train.groupby(['race_key', 'pla']).first().reset_index()
for ep in range(args.epoch):
    t0 = time.time()
    ep_loss = []
    train_quick, train_slow = pairwise_sampler(train, args.sampling_perc)
    Xs1, _ = get_feats(train_quick, numerical_cols, y_col, use_cuda)
    Xs2, _ = get_feats(train_slow, numerical_cols, y_col, use_cuda)
        
    for batch_data in horse_pairwise_data_loader(Xs1, Xs2, args.batch_size, shuffle=True):
        X1, X2 = batch_data
        X1, X2 = split_cols(X1), split_cols(X2)
        model.zero_grad()

        y_pred1 = model(*X1)
        y_pred2 = model(*X2)

        loss = PairwiseLoss(y_pred1, y_pred2)
        loss.mean().backward()
        opt.step()

        ep_loss.append(loss.data.to(device('cpu')).tolist()[0])
#         print(ep_loss)
        
    train_loss_by_ep.append(np.mean(ep_loss))
    
    # compute AP
    val_ap_by_ep.append(computeAP(val, model, way='max', use_cuda=use_cuda))
    test_ap_by_ep.append(computeAP(test, model, way='max', use_cuda=use_cuda))
    
    t1 = time.time()
    print(f'[{round(t1-t0, 3)}s] Iter={ep}, train loss={round(train_loss_by_ep[-1], 3)} [VAL] AP={round(val_ap_by_ep[-1], 3)}; [TEST] AP={round(test_ap_by_ep[-1], 3)};')