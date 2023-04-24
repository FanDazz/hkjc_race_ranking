import pandas as pd
import warnings
warnings.filterwarnings('ignore')

"""
path
scale
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


file_root = './horse/data/perform_clean.csv'
y_cols = ['is_champ', 'pla', 'finish_time', 'speed']
date_cols = ['race_key', 'race_date']
scaling_cols = ['distance', 'race_money', 'act_wt', 'declare_horse_wt']
get_x_cols = lambda x: [col for col in x if (col not in y_cols) and (col not in date_cols)]

class DataSet:
    def __init__(self, path=file_root, scaling=True) -> None:
        self.data = pd.read_csv(path, sep=',', encoding='utf-8')
        self.data = self._pre_cleanse()
        self.x_cols = get_x_cols(self.data.columns)
        self.data = self.categorization()
        if scaling:
            self.scaling_info = {}
            self.data = self.scaling(scaling_cols)

    def _pre_cleanse(self, ):
        self.data['speed'] = self.data['distance']/self.data['finish_time']
        self.data['is_champ'] = self.data['pla'].apply(lambda x: 1 if x==1 else 0)
        return self.data

    def categorization(self, ):
        self.data['course_type'] = self.data['course_type'].replace({'全天候跑道':0, '草地':1})
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

        return self.data
    
    def scaling(self, scaling_cols):
        for col in scaling_cols:
            self.scaling_info[col] = {}
            self.scaling_info[col]['mean'], self.scaling_info[col]['std'], self.data[col] = zscore_standarlization(self.data[col])

        return self.data

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
