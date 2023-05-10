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

def train_val_test_split(data, col, perc=[0.8, 0.1, 0.1]):
    time_map = data[col].drop_duplicates()
    n_dates = time_map.__len__()
    cut_point1 = int(perc[0]*n_dates)
    cut_point2 = int((perc[0]+perc[1])*n_dates)

    train_date, val_date, test_date = time_map[:cut_point1], time_map[cut_point1:cut_point2], time_map[cut_point2:]
    train, val, test = data.merge(train_date, how='inner'), data.merge(val_date, how='inner'), data.merge(test_date, how='inner')

    return train, val, test

file_root = './horse/data/perform_full_feature.csv'
y_cols = ['is_champ', 'pla', 'finish_time', 'speed', 'is_champ', 'horse_is_champ', 'horse_is_place', 'horse_is_top4', 'jockey_is_champ', 'jockey_is_place', 'jockey_is_top4', 'trainer_is_champ', 'trainer_is_place', 'trainer_is_top4', 'win_odds']
date_cols = ['race_key', 'race_date']
dummy_cols = ['dr', 'dr_ix', 'field_going', 'jockey' ,'horse', 'trainer']
best_feat = [
    'race_money',
    'horse_bestperform_h',
    'horse_champ_rate_h',
    'horse_champs_h',
    'horse_life_time_d',
    'horse_place_m',
    'horse_top4_rate_y',
    'horse_top4_h',
    'horse_top4_last5',
    'jockey_champ_rate_h',
    'jockey_champ_last5',
    'jockey_champ_y',
    'jockey_champ_rate_y',
    'jockey_elo',
    'jockey_top4_rate_y',
    'jockey_top4_m',
    'jockey_top4_rate_m',
    'trainer_champ_rate_h',
    'trainer_champ_rate_y',
    'trainer_place_rate_h',
    'trainer_top4_rate_y',
    'trainer_top4_rate_m'
]

get_x_cols = lambda x: [col for col in x if (col not in y_cols+date_cols)]
get_scaling_cols = lambda x: [col for col in x if (col not in y_cols+date_cols+dummy_cols)]

class DataSet:
    """ Call Dataset from src.
        
        [Usage] Call best subset of feature.
        >>> dataset = DataSet(scaling=False, do_categorization=False, use_best_feats=True)

        [Usage] Do train, validation, test split.
        >>> train, val, test = dataset.train_val_test_split(perc=[0.8, 0.1, 0.1])
    """
    def __init__(self, path=file_root, scaling=True
                                     , do_categorization=False
                                     , use_best_feats=True) -> None:
        self.categorize=do_categorization
        self.data = pd.read_csv(path, sep=',', encoding='utf-8')
        self.data = self._pre_cleanse()
        self.x_cols = get_x_cols(self.data.columns)
        if do_categorization:
            self.do_categorization()
        if scaling:
            self.scaling_info = {}
            scaling_cols = get_scaling_cols(self.data.columns)
#             print(f'Scaling cols: {scaling_cols}')
            self.do_scaling(scaling_cols)
        if use_best_feats:
            self.do_best_feats()
        else:
            self.do_not_use_best_feats()


    def _pre_cleanse(self, ):
        self.data['speed'] = self.data['distance']/self.data['finish_time']
        self.data['is_champ'] = self.data['pla'].apply(lambda x: 1 if x==1 else 0)
        self.data['course_type'] = self.data['course_type'].replace({'全天候跑道':0, '草地':1})
        return self.data

    def do_categorization(self, do_dummy=False
                              , categ_cols=['dr_ix', 'horse', 'jockey', 'trainer', 'field_going']):
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
        return self        
    
    def do_scaling(self, scaling_cols):
        for col in scaling_cols:
            self.scaling_info[col] = {}
            self.scaling_info[col]['mean'], self.scaling_info[col]['std'], self.data[col] = zscore_standarlization(self.data[col])

        return self
    
    def do_best_feats(self, target='is_champ'):
        categ_cols = ['dr', 'field_going', 'jockey', 'horse', 'trainer']
        if self.categorize:
            categ_cols += ['dr_ix']
        cols = date_cols + categ_cols + best_feat + [target, 'pla']
        self.data = self.data[cols]
        
        return self
    
    def do_not_use_best_feats(self, target='is_champ'):
        categ_cols = ['dr', 'field_going', 'jockey', 'horse', 'trainer']
        numeric_cols = get_scaling_cols(self.data.columns)
        if self.categorize:
            categ_cols += ['dr_ix']
        cols = date_cols + numeric_cols + categ_cols + [target, 'pla']
        self.data = self.data[cols]
        
        return self
    
    def my_train_val_test_split(self, perc=[0.8, 0.1, 0.1]):
        train, val, test = train_val_test_split(self.data, 'race_date', perc)
        
        return train, val, test
