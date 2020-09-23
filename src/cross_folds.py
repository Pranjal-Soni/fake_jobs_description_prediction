# -*- coding: utf-8 -*-

import config
import pandas as pd
from sklearn import model_selection


def create_folds(df):
     #create a k-fold column and initialise it with -1
    df["kfold"] = -1
    

    
    #intialise kfold class
    kf = model_selection.StratifiedKFold(n_splits=5)
    
    #filling kfold columns
    for f,(t_,v_) in enumerate(kf.split(X=df,y=df.fraudulent)):
        df.loc[v_,'kfold'] = f
    
    return df

if __name__ == "__main__":
    #read the dataset file
    df = pd.read_csv(config.TRAIN_DATA)

    #suffling the dataset
    df = df.sample(frac=1).reset_index(drop = True)
    
    df = create_folds(df)
    
    df.to_csv('../inputs/train_folds.csv',index=False)
    