# -*- coding: utf-8 -*-

import config
import joblib
import pandas as pd
from sklearn import svm
from sklearn import metrics,model_selection

def score(df,fold):

    df_train = df[df["kfold"] != fold].reset_index(drop=True)
    df_valid = df[df["kfold"] == fold].reset_index(drop=True)
    
    x_train = df_train.drop('fraudulent',axis=1).values
    y_train = df_train.fraudulent.values
    
    x_valid = df_valid.drop('fraudulent',axis=1).values
    y_valid = df_valid.fraudulent.values
    
    model = svm.SVC(C=10, gamma=0.001, kernel='linear')
    model.fit(x_train,y_train)
    y_pred = model.predict(x_valid)
    
    print(metrics.confusion_matrix(y_valid,y_pred))
    
    return metrics.roc_auc_score(y_valid,y_pred)
    
def best_parameter(df):

    X = df.drop(['kfold','fraudulent'],axis=1)
    Y = df.fraudulent.values
        
    hyper = {'C':[0.001,0.01,10,100,1000],
             'gamma':[0.001,0.01,10,100,1000],
             'kernel':['linear']
            }
    
    gd=model_selection.GridSearchCV(estimator=svm.SVC(),
                                    scoring = 'roc_auc',
                                    param_grid=hyper,
                                    n_jobs=-1,
                                    verbose=True)
    
    gd.fit(X,Y)
    return gd
    
#SVC(C=10, gamma=0.001, kernel='linear')

if __name__ == "__main__":
    df = pd.read_csv(config.KFOLD_TRAIN_DATA)
    test_df = pd.read_csv(config.TEST_DATA)
    
    gd = best_parameter(df)

    for fold in range(0,5):
        print(score(df,fold))
    
    #x_test = test_df.drop('fraudulent',axis=1)
    #y_test = test_df.fraudulent.values

