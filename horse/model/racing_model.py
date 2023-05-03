import torch.nn as nn
from torch.nn.functional import logsigmoid
import torch

# loss
def MSELoss(input, target):
    return (target-input)**2

def BCELoss(input, target):
    return nn.BCEWithLogitsLoss()(input, target)

def LogSigmoidLoss(pred1, pred2):
    return -logsigmoid(pred1-pred2)

# models
class _BaseModel(nn.Module):
    def __init__(self, n_dr, n_field
                     , n_jockey, n_horse, n_trainer
                     , n_num_feats, k_dim_field, k_dim_id
                     , need_activation=True) -> None:
        super(_BaseModel, self).__init__()
        self.out_dim = self.compute_out_dim(n_num_feats, k_dim_field, k_dim_id)
        self.activation = torch.sigmoid if need_activation else torch.nn.Identity()
        # init embedding for ids
        self.emb_dr = nn.Embedding(n_dr, k_dim_field)
        self.emb_field = nn.Embedding(n_field, k_dim_field)
        self.emb_jockey = nn.Embedding(n_jockey, k_dim_id)
        self.emb_horse = nn.Embedding(n_horse, k_dim_id)
        self.emb_trainer = nn.Embedding(n_trainer, k_dim_id)
        # output layer
        self.relu = nn.ReLU()
        self.Linear = nn.Linear(self.out_dim, 1)
        # init all dims
        nn.init.kaiming_normal_(self.Linear.weight)
        nn.init.kaiming_normal_(self.emb_dr.weight)
        nn.init.kaiming_normal_(self.emb_field.weight)
        nn.init.kaiming_normal_(self.emb_jockey.weight)
        nn.init.kaiming_normal_(self.emb_horse.weight)
        nn.init.kaiming_normal_(self.emb_trainer.weight)        

    def forward(self, x, dr, field, jockey, horse, trainer):
        pass
    
    def compute_out_dim(self, n_num_feats, k_dim_field, k_dim_id):
        pass


class LinEmbConcat(_BaseModel):
    """ Simple Concat + Linear txfm for all embeddings """
    def __init__(self, n_dr, n_field
                     , n_jockey, n_horse, n_trainer
                     , n_num_feats, k_dim_field, k_dim_id
                     , need_activation) -> None:
        super(LinEmbConcat, self).__init__(n_dr, n_field
                                            , n_jockey, n_horse, n_trainer
                                            , n_num_feats, k_dim_field, k_dim_id
                                            , need_activation)

    def forward(self, x, dr, field, jockey, horse, trainer):
        # lookup layer
        emb_d = self.emb_dr(dr)
        emb_f = self.emb_field(field)
        emb_j = self.emb_jockey(jockey)
        emb_h = self.emb_horse(horse)
        emb_t = self.emb_trainer(trainer)
        # out layer
        out = torch.concat([x, emb_d, emb_f, emb_j, emb_h, emb_t], 1)
        out = self.Linear(self.relu(out))
        out = self.activation(out)
        return out
    
    def compute_out_dim(self, n_num_feats, k_dim_field, k_dim_id):
        return n_num_feats + 2*k_dim_field + 3*k_dim_id


class LinEmbDotProd(_BaseModel):
    """ Dot Product for interactions """
    def __init__(self, n_dr, n_field
                     , n_jockey, n_horse, n_trainer
                     , n_num_feats, k_dim_field, k_dim_id
                     , need_activation) -> None:
        super(LinEmbDotProd, self).__init__(n_dr, n_field
                                            , n_jockey, n_horse, n_trainer
                                            , n_num_feats, k_dim_field, k_dim_id
                                            , need_activation)
        self.Linear_field = nn.Linear(k_dim_field, 1)
        self.Linear_dr = nn.Linear(k_dim_field, 1)
        # init all dims
        nn.init.kaiming_normal_(self.Linear_field.weight)
        nn.init.kaiming_normal_(self.Linear_dr.weight)


    def forward(self, x, dr, field, jockey, horse, trainer):
        # lookup layer
        emb_d = self.emb_dr(dr)
        emb_f = self.emb_field(field)
        emb_j = self.emb_jockey(jockey)
        emb_h = self.emb_horse(horse)
        emb_t = self.emb_trainer(trainer)
        # compute, interaction layer
        d_val = self.Linear_dr(emb_d)
        f_val = self.Linear_field(emb_f)
        hj_val = torch.matmul(emb_h, emb_j.T).sum(1).unsqueeze(1)
        ht_val = torch.matmul(emb_h, emb_t.T).sum(1).unsqueeze(1)
        # out layer
        out = torch.concat([x, d_val, f_val, hj_val, ht_val],1)
        out = self.Linear(self.relu(out))
        out = self.activation(out)
        return out
    
    def compute_out_dim(self, n_num_feats, k_dim_field, k_dim_id):
        return n_num_feats + 4
    

class LinEmbElemProd(_BaseModel):
    """ Element wise multiplication of embs, and linear comb """
    def __init__(self, n_dr, n_field
                     , n_jockey, n_horse, n_trainer
                     , n_num_feats, k_dim_field, k_dim_id
                     , need_activation) -> None:
        super(LinEmbElemProd, self).__init__(n_dr, n_field
                                             , n_jockey, n_horse, n_trainer
                                             , n_num_feats, k_dim_field, k_dim_id
                                             , need_activation)  

    def forward(self, x, dr, field, jockey, horse, trainer):
        # lookup
        emb_d = self.emb_dr(dr)
        emb_f = self.emb_field(field)
        emb_j = self.emb_jockey(jockey)
        emb_h = self.emb_horse(horse)
        emb_t = self.emb_trainer(trainer)
        # interaction layer
        hj_val = torch.mul(emb_h, emb_j)
        ht_val = torch.mul(emb_h, emb_t)
        # out Layer
        out = torch.concat([x, emb_d, emb_f, hj_val, ht_val], 1)
        out = self.Linear(self.relu(out))
        return self.activation(out)
    
    def compute_out_dim(self, n_num_feats, k_dim_field, k_dim_id):
        return n_num_feats + 2*k_dim_field + 2*k_dim_id
    

class EmbMLP(_BaseModel):
    """ Pass all data through MLP for LinEmbElemProd """
    def __init__(self, n_dr, n_field
                     , n_jockey, n_horse, n_trainer
                     , n_num_feats, k_dim_field, k_dim_id
                     , need_activation
                     , num_layers, p_dropout=0.1, layer_size_reduction=0.5) -> None:
        super(EmbMLP, self).__init__(n_dr, n_field
                                     , n_jockey, n_horse, n_trainer
                                     , n_num_feats, k_dim_field, k_dim_id
                                     , need_activation)  
        # MLP
        feat_dim = n_num_feats + 2*k_dim_field + 3*k_dim_id
        MLP_sizes = [int(feat_dim*(layer_size_reduction**i)) for i in range(num_layers+1)]
        MLP_Layer=[]
        for i in range(num_layers):
            MLP_Layer.append(nn.Dropout(p_dropout))
            MLP_Layer.append(nn.Linear(MLP_sizes[i], MLP_sizes[i+1]))
            MLP_Layer.append(nn.ReLU())  
        self.MLP_Layer = nn.Sequential(*MLP_Layer)
        self.out_dim = MLP_sizes[-1]
        self.Linear = nn.Linear(self.out_dim, 1)
        for layer in self.MLP_Layer:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)

    def forward(self, x, dr, field, jockey, horse, trainer):
        # lookup
        emb_d = self.emb_dr(dr)
        emb_f = self.emb_field(field)
        emb_j = self.emb_jockey(jockey)
        emb_h = self.emb_horse(horse)
        emb_t = self.emb_trainer(trainer)
        # interaction layer
#         hj_val = torch.mul(emb_h, emb_j)
#         ht_val = torch.mul(emb_h, emb_t)
        out = torch.concat([x, emb_d, emb_f, emb_j, emb_h, emb_t], 1)
        # MLP layer
        out = self.MLP_Layer(out)
        # out layer
        out = self.Linear(out)
        out = self.activation(out)
        return out

    def compute_out_dim(self, n_num_feats, k_dim_field, k_dim_id):
        return 1