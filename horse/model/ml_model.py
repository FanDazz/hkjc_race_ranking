import warnings
warnings.filterwarnings('ignore')

class HKJC_models():
    def __init__(self,param):
        self.param = param
        self.model_dict = {
            "logistic": self.logistic_regression(),
            "dtc": self.decsion_classifier(),
            "rfc": self.rf_classifier(),
            "adc": self.ada_classifier(),
            "ridge": self.ridge_regression(),
            "dtr": self.decision_regression(),
            "rfr": self.rf_regression(),
            "adr": self.ada_regression(),
            "xgbc": self.xgb_classifier(),
            "xgbr": self.xgb_regression()
        }


    def logistic_regression(self):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(penalty=self.param["penalty"],
                                   random_state=self.param["random_state"],
                                   solver=self.param["solver"],
                                   C=self.param["C"],
                                   max_iter=self.param["max_iter"])
        return model

    def decsion_classifier(self):
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier(criterion=self.param["criterion"],
                                       random_state=self.param["random_state"],
                                       splitter=self.param["splitter"],
                                       max_depth=self.param["max_depth"],
                                       min_samples_leaf=self.param["min_samples_leaf"],
                                       min_samples_split=self.param["min_samples_split"])

        return model

    def rf_classifier(self):
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(criterion=self.param["criterion"],
                                       random_state=self.param["random_state"],
                                       max_depth=self.param["max_depth"],
                                       min_samples_leaf=self.param["min_samples_leaf"],
                                       min_samples_split=self.param["min_samples_split"])
        return model

    def ada_classifier(self):
        from sklearn.ensemble import AdaBoostClassifier
        model = AdaBoostClassifier(n_estimators=self.param["n_estimators"],
                                   learning_rate=self.param["learning_rate"],
                                   algorithm=self.param["algorithm"],
                                   random_state=self.param["random_state"]
        )
        return model

    def ridge_regression(self):
        from sklearn.linear_model import Ridge
        model = Ridge(alpha=self.param["alpha"],
                      solver=self.param["solver"],
                      random_state=self.param["random_state"])
        return model

    def decision_regression(self):
        from sklearn.tree import DecisionTreeRegressor
        model = DecisionTreeRegressor(criterion=self.param["criterion"],
                                      random_state=self.param["random_state"],
                                      splitter=self.param["splitter"],
                                      max_depth=self.param["max_depth"],
                                      min_samples_leaf=self.param["min_samples_leaf"],
                                      min_samples_split=self.param["min_samples_split"])
        return model

    def rf_regression(self):
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(criterion=self.param["criterion"],
                                       random_state=self.param["random_state"],
                                       max_depth=self.param["max_depth"],
                                       min_samples_leaf=self.param["min_samples_leaf"],
                                       min_samples_split=self.param["min_samples_split"])
        return model

    def ada_regression(self):
        from sklearn.ensemble import AdaBoostRegressor
        model = AdaBoostRegressor(n_estimators=self.param["n_estimators"],
                                   learning_rate=self.param["learning_rate"],
                                   loss=self.param["loss"],
                                   random_state=self.param["random_state"])
        return model

    def xgb_classifier(self):
        from xgboost.sklearn import XGBClassifier

        xgb_classifier = XGBClassifier(
            tree_method=self.param["tree_method"],
            learning_rate=self.param["learning_rate"],
            n_estimators=self.param["n_estimators"],
            max_depth=self.param["max_depth"],
            min_child_weight=self.param["min_child_weight"],
            subsample=self.param["subsample"],
            colsample_bytree=self.param["colsample_bytree"],
            reg_alpha=self.param["reg_alpha"],
            reg_lambda=self.param["reg_lambda"],
            objective=self.param["objective"],
            eval_metric=self.param["eval_metric"])

        return xgb_classifier

    def xgb_regression(self):
        from xgboost.sklearn import XGBRegressor

        xgb_regressor = XGBRegressor(
            tree_method=self.param["tree_method"],
            learning_rate=self.param["learning_rate"],
            n_estimators=self.param["n_estimators"],
            max_depth=self.param["max_depth"],
            min_child_weight=self.param["min_child_weight"],
            subsample=self.param["subsample"],
            colsample_bytree=self.param["colsample_bytree"],
            reg_alpha=self.param["reg_alpha"],
            reg_lambda=self.param["reg_lambda"],
            eval_metric=self.param["eval_metric"])

        return xgb_regressor

    def comupte_champ(self, df, model, x_cols, kind='clf', way='max'):
        """
        :param df: performance dataframe
        :param model: classifier model
        :param x_cols: indexes of dataframe
        :param kind: reg: regression model,clf: calssification model
        :return: prediciton result
        """
        X = df[x_cols]
        result = df[['race_key', 'dr']]

        if kind == 'clf':
            result['win'] = model.predict_proba(X)[:, 1]

        elif kind == 'reg':
            result['win'] = model.predict(X)

        if way == 'max':
            return result.groupby(['race_key']) \
                .apply(lambda x: x[x['win'] == x['win'].max()]) \
                .reset_index(drop=True)[['race_key', 'dr']]
        else:
            return result.groupby(['race_key']) \
                .apply(lambda x: x[x['win'] == x['win'].min()]) \
                .reset_index(drop=True)[['race_key', 'dr']]

    def save_model(self,model,model_type):
        import joblib
        import os
        path = 'saved_models/'+model_type+'/'
        try:
            os.makedirs(path)
        except:
            pass
        joblib.dump(model, 'saved_models/'+model_type+'/'+model_type+'.pkl')
        print("Model Saved!")

    def load_model(self,model,model_type):
        import joblib
        print("Loading model:",model_type)
        return joblib.load(model,'saved_models/'+model_type+'/'+model_type+'.pkl')

    def cal_result(self, model_name, model_target, X, y, dm_perform_val, x_cols):

        import time
        
        def AveragePrecision(input, target):
            merge_df = input.merge(target, on='race_key', how='left')
            
            return (merge_df.iloc[:, 1]==merge_df.iloc[:, 2]).sum()/merge_df.shape[0]
        
        def racing_champ(df):
            return df[df['pla']==1][['race_key', 'dr']]

        if model_target == "is_champ":
            kind = 'clf'
            way = 'max'
            val_target = racing_champ(dm_perform_val)
            model = self.model_dict[model_name]
            # train on training set
            t0 = time.time()
            # train on training set
            model.fit(X, y)
            t1 = time.time()
            print(f'[{round(t1 - t0, 3)} s] DONE {model_name}.')
            # eval on validation set
            pred = self.comupte_champ(dm_perform_val, model, x_cols, kind, way)
            print(f'AP for {model_name}: {round(AveragePrecision(input=pred, target=val_target), 4)}')
            self.save_model(model, model_name)

        else:
            kind = 'reg'
            way = 'min'
            val_target = racing_champ(dm_perform_val)
            model = self.model_dict[model_name]
            # train on training set
            t0 = time.time()
            # train on training set
            model.fit(X, y)
            t1 = time.time()
            print(f'[{round(t1 - t0, 3)} s] DONE {model_name}.')
            # eval on validation set
            pred = self.comupte_champ(dm_perform_val, model, x_cols, kind, way)
            print(f'AP for {model_name}: {round(AveragePrecision(input=pred, target=val_target), 4)}')
            self.save_model(model,model_name)
