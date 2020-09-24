# -*- coding: utf-8 -*-

import config
import joblib
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import metrics,model_selection

def score(df,model):
    
    for fold in range(0,5):
        scores = []
        df_train = df[df["kfold"] != fold].reset_index(drop=True)
        df_valid = df[df["kfold"] == fold].reset_index(drop=True)
        
        x_train = df_train.drop(['kfold','fraudulent'],axis=1)
        y_train = df_train.fraudulent.values
        
        
        x_valid = df_valid.drop(['kfold','fraudulent'],axis=1).values
        y_valid = df_valid.fraudulent.values
        
        model.fit(x_train,y_train)
        y_pred = model.predict(x_valid)
        
        scores.append(metrics.roc_auc_score(y_valid,y_pred))
        #print(metrics.confusion_matrix(y_valid,y_pred))

    score = np.mean(scores)   
    return score
    
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
    
    #importing the datasets
    df = pd.read_csv(config.KFOLD_TRAIN_DATA)
    os_df = pd.read_csv(config.OVERSAMPLED_TRAIN_DATA)
    test_df = pd.read_csv(config.TEST_DATA)
       
    #calculate intial score without hyperparameter tuning 
    #using K-stratified K fold
    model = svm.SVC()
    intial_score = score(df,model)
    print(f"Intial roc_auc Score is : {intial_score}")
    
    #calculter score for test data
    x_test = test_df.drop('fraudulent',axis=1).values
    y_test = test_df.fraudulent.values
    y_pred = model.predict(x_test)
    test_score = metrics.roc_auc_score(y_test,y_pred)
    print(f'Intial Test score : {test_score}')
    
    #tune hyperparameter and getting best parameters
    params = best_parameter(os_df)
    print(f"Best parameter are {params}")

    #training our training dataset on the best hyperparameter 
    model = svm.SVC(**params)
    
    #fit the model for oversampled data
    X = os_df.drop('fraudulent',axis=1).values
    y = os_df.fraudulent.values
    model.fit(X,y)
    
    #calculter score for test data
    y_pred = model.predict(x_test)
    test_score = metrics.roc_auc_score(y_test,y_pred)
    print(f'Test score : {test_score}')

    #saving the model
    joblib.dump(model, '../models/svm_model.pkl') 

