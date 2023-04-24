
def racing_champ(df):
    return df[df['pla']==1][['race_key', 'dr']]


def AveragePrecision(input, target):
    merge_df = input.merge(target, on='race_key', how='left')
    
    return (merge_df.iloc[:, 1]==merge_df.iloc[:, 2]).sum()/merge_df.shape[0]


remove_odds = lambda x: [col for col in x if col != 'win_odds']

def comupte_champ(df, model, x_cols, kind='clf'):
    X = df[x_cols]
    result = df[['race_key', 'dr']]

    if kind=='clf':
        result['win'] = model.predict_proba(X)[:, 1]
    elif kind=='reg':
        result['win'] = model.predict(X)

    return result.groupby(['race_key']) \
                .apply(lambda x: x[x['win']==x['win'].max()]) \
                .reset_index(drop=True)[['race_key', 'dr']]