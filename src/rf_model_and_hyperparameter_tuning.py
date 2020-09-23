# -*- coding: utf-8 -*-

import config
import pandas as pd
from sklearn import ensemble
from sklearn import metrics


def optimize(df,fold):

    df_train = df[df["kfold"] != fold].reset_index(drop=True)
    df_valid = df[df["kfold"] == fold].reset_index(drop=True)
    
    x_train = df_train.drop('fraudulent',axis=1).values
    y_train = df_train.fraudulent.values
    
    x_valid = df_valid.drop('fraudulent',axis=1).values
    y_valid = df_valid.fraudulent.values
    
    model = ensemble.RandomForestClassifier()
    model.fit(x_train,y_train)
    y_pred = model.predict(x_valid)
    
    print(metrics.confusion_matrix(y_valid,y_pred))
    
    return metrics.roc_auc_score(y_valid,y_pred)
    
    
    
    
if __name__ == "__main__":
    df = pd.read_csv(config.KFOLD_TRAIN_DATA)
    
    for fold in range(0,5):
        print(optimize(df,fold))
    

    
    