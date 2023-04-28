import pandas as pd
import warnings
warnings.filterwarnings('ignore')

"""
path
scale
mapping_or_dummy
"""

def col2ix(df_col:pd.Series):
    """ Mapping from column's item to indecies.

        Returns:
            ix2item: dict
                with key as index, value is the hashed item
            item2ix: dict
                with key as item, value as its index
    """
    ix = list(range(df_col.nunique()))
    item = df_col.unique().tolist()

    return dict(zip(ix, item)), dict(zip(item, ix))


def zscore_standarlization(col):
    """ Z-score standarization for further penalization """
    mean = col.mean()
    std = col.std()
    col_standarlize = (col-mean)/std 
    return mean, std, col_standarlize


file_root = './horse/data/perform_full.csv'
y_cols = ['is_champ', 'pla', 'finish_time', 'speed']
date_cols = ['race_key', 'race_date']
scaling_cols = ['distance', 'race_money', 'act_wt', 'declare_horse_wt']
best_feat = [
    'distance',
     'race_money',
     'act_wt',
     'declare_horse_wt',
     'horse_bestperform_h',
     'horse_elo',
     'horse_elo_course',
     'horse_last_comp_days',
     'horse_life_time_d',
     'horse_racenum_y',
     'horse_showups',
     'horse_top4_rate_y',
     'jockey_champ_h',
     'jockey_champ_last5',
     'jockey_champ_y',
     'jockey_champ_m',
     'jockey_champ_d',
     'jockey_champ_rate_d',
     'jockey_elo',
     'jockey_elo_course',
     'jockey_last_comp_days',
     'jockey_life_time_d',
     'jockey_place_h',
     'jockey_place_y',
     'jockey_racenum_y',
     'jockey_racenum_m',
     'jockey_racenum_d',
     'jockey_showups',
     'jockey_top4_h',
     'jockey_top4_y',
     'jockey_top4_d',
     'trainer_champ_h',
     'trainer_champ_last5',
     'trainer_champ_y',
     'trainer_champ_rate_y',
     'trainer_champ_m',
     'trainer_champ_d',
     'trainer_champ_rate_d',
     'trainer_elo',
     'trainer_elo_course',
     'trainer_last_comp_days',
     'trainer_life_time_d',
     'trainer_place_h',
     'trainer_place_y',
     'trainer_place_rate_y',
     'trainer_racenum_y',
     'trainer_racenum_m',
     'trainer_racenum_d',
     'trainer_showups',
     'trainer_top4_h',
     'trainer_top4_y',
     'trainer_top4_rate_y',
     'trainer_top4_m',
     'trainer_top4_d'
]

get_x_cols = lambda x: [col for col in x if (col not in y_cols) and (col not in date_cols)]

class DataSet:
    def __init__(self, path=file_root, scaling=True, do_categorization=False, use_best_feats=True) -> None:
        self.data = pd.read_csv(path, sep=',', encoding='utf-8')
        self.data = self._pre_cleanse()
        self.x_cols = get_x_cols(self.data.columns)
        if do_categorization:
            self.do_categorization()
        if scaling:
            self.scaling_info = {}
            self.do_scaling(scaling_cols)
        if use_best_feats:
            self.do_best_feats()

    def _pre_cleanse(self, ):
        self.data['speed'] = self.data['distance']/self.data['finish_time']
        self.data['is_champ'] = self.data['pla'].apply(lambda x: 1 if x==1 else 0)
        self.data['course_type'] = self.data['course_type'].replace({'全天候跑道':0, '草地':1})
        return self.data

    def do_categorization(self, do_dummy=False
                           , categ_cols=['dr_ix', 'horse', 'jockey', 'trainer', 'field_going']):
        if not do_dummy:
            # mapping dict
            self.ix2dr, self.dr2ix = col2ix(self.data['dr'])
            self.ix2field, self.field2ix = col2ix(self.data['field_going'])
            self.ix2jockey, self.jockey2ix = col2ix(self.data['jockey'])
            self.ix2horse, self.horse2ix = col2ix(self.data['horse'])
            self.ix2trainer, self.trainer2ix = col2ix(self.data['trainer'])
            # perform after mapping
            self.data['dr_ix'] = self.data['dr'].apply(lambda x: self.dr2ix[x])
            self.data['field_going'] = self.data['field_going'].apply(lambda x: self.field2ix[x])
            self.data['jockey'] = self.data['jockey'].apply(lambda x: self.jockey2ix[x])
            self.data['horse'] = self.data['horse'].apply(lambda x: self.horse2ix[x])
            self.data['trainer'] = self.data['trainer'].apply(lambda x: self.trainer2ix[x])
        else:
            self.data = pd.get_dummies(self.data, columns=categ_cols)

        return self
    
    def do_scaling(self, scaling_cols):
        for col in scaling_cols:
            self.scaling_info[col] = {}
            self.scaling_info[col]['mean'], self.scaling_info[col]['std'], self.data[col] = zscore_standarlization(self.data[col])

        return self
    
    def do_best_feats(self, target='is_champ'):
        cols = date_cols + best_feat + [target, 'pla']
        self.data = self.data[cols]
        return self
    
    def train_val_test_split(self, perc=[0.8, 0.1, 0.1]):

        def _train_test_split(data, col, perc=[0.8, 0.1, 0.1]):
            time_map = data[col].drop_duplicates()
            n_dates = time_map.__len__()
            cut_point1 = int(perc[0]*n_dates)
            cut_point2 = int((perc[0]+perc[1])*n_dates)

            train_date, val_date, test_date = time_map[:cut_point1], time_map[cut_point1:cut_point2], time_map[cut_point2:]
            train, val, test = data.merge(train_date, how='inner'), data.merge(val_date, how='inner'), data.merge(test_date, how='inner')
            
            return train, val, test

        train, val, test = _train_test_split(self.data, 'race_date', perc)
        
        return train, val, test
