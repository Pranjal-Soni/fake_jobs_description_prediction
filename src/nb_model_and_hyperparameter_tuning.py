# -*- coding: utf-8 -*-

import config
import pandas as pd
import numpy as np
from sklearn import naive_bayes
from sklearn import metrics
from sklearn import model_selection

def score(df):
    
    for fold in range(0,10):
        scores = []
        df_train = df[df["kfold"] != fold].reset_index(drop=True)
        df_valid = df[df["kfold"] == fold].reset_index(drop=True)
        
        x_train = df_train.drop(['kfold','fraudulent'],axis=1)
        y_train = df_train.fraudulent.values
        
        
        x_valid = df_valid.drop(['kfold','fraudulent'],axis=1).values
        y_valid = df_valid.fraudulent.values
        
        model = naive_bayes.GaussianNB(var_smoothing= 0.0533669923120631)
        model.fit(x_train,y_train)
        y_pred = model.predict(x_valid)
        
        scores.append(metrics.roc_auc_score(y_valid,y_pred))
        print(metrics.confusion_matrix(y_valid,y_pred))

    score = np.mean(scores)   
    return score
    


def best_parameter(df):

    X = df.drop(['kfold','fraudulent'],axis=1)
    Y = df.fraudulent.values
        
    hyper = { 
            'var_smoothing': np.logspace(0,-9, num=100)
            }
    
    gd=model_selection.GridSearchCV(estimator=naive_bayes.GaussianNB(),
                                    scoring = 'roc_auc',
                                    param_grid=hyper,
                                    n_jobs=-1,
                                    verbose=True)
    
    gd.fit(X,Y)
    return gd
    
if __name__ == "__main__":
    df = pd.read_csv(config.KFOLD_TRAIN_DATA)
    test_df = pd.read_csv(config.TEST_DATA)
    
    os_df = pd.read_csv(config.OVERSAMPLED_TRAIN_DATA)
    os_df = os_df.sample(frac=1).reset_index(drop=True)
    kf = model_selection.KFold(n_splits=10)
    
    #filling kfold columns
    for f,(t_,v_) in enumerate(kf.split(X=os_df,y=os_df.fraudulent)):
        os_df.loc[v_,'kfold'] = f
        
    
    print(score(os_df))
        
    #gd = best_parameter(df)
    
    x_test = test_df.drop('fraudulent',axis=1)
    y_test = test_df.fraudulent.values
        
    model = naive_bayes.GaussianNB(var_smoothing= 0.0533669923120631)
    model.fit()
    y_pred = model.predict(x_test)
    score = metrics.roc_auc_score(y_test,y_pred)
    print(f'Test score : {score}')

    
    
