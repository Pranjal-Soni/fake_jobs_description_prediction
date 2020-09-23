# -*- coding: utf-8 -*-

import config
import pandas as pd
from sklearn.utils import resample



#oversampling the dataset to hadle imbalanced dataset
def oversampling(df,target_name):
    
    
    
    # Separating classes
    fraud = df[df[target_name] == 1]
    not_fraud = df[df[target_name] == 0]
    
    #oversample the dataset with fraudulent columns
    oversample = resample(fraud, 
                       replace=True, 
                       n_samples=len(not_fraud),
                       random_state=42)
    
    # Returning to new training set
    oversample_train = pd.concat([not_fraud, oversample])
    oversample_train[target_name].value_counts(normalize=True)
    
    # Separate oversampled data into X and y sets
    oversample_x_train = oversample_train.drop('fraudulent', axis=1)
    oversample_y_train = oversample_train[target_name]
    
    df = pd.concat([oversample_x_train, oversample_y_train],axis=1)
    return df


if __name__ == "__main__":
    #read the dataset file
    df = pd.read_csv(config.TRAIN_DATA)

    #suffling the dataset
    df = df.sample(frac=1).reset_index(drop = True)
    
    df = oversampling(df,'fraudulent')
    
    df.to_csv('../inputs/oversampled_train.csv',index=False)