# Fake News Description Prediction

This is a data science project and problem is to identify given job description is fraudulent or not. The dataset is taken from Kaggel and it consists meta-information about 18K job descriptions. We deal with numerical and text features in this project. The project consists of data preprocessing, data analysis, feature engineering, handling imbalanced dataset and hyperparameter tuning.

#### -- Project Status: [Completed]

## Objective 
In the current scenario, there is too much job posting online and we got daily mail for this and that job posting in an organisation or a company. We look at it and it seems that this is the perfect job for you. But it is also possible that there is no such job and the job description is fake. They are trying to collect your data or personal information. So the project is on fake job description prediction is to predict whether the job description is given is genuine or this is for some malicious and job is fake and they are trying to just to fool you.

### Partner
* Sajal Sharma
* Contact : ssajal1998@gmail.com

### Methods Used
* Data Cleaning
* Feature engineering
* Data Visulisation
* Machine Learning
* Hyperparameter Tuning
### Technologies 
* Python
* Jupyter Notebook, Spyder
* Pandas , Sklearn, Numpy
* Matplotlib, Seaborn

## Project Description
#### Dataset :
The data is collected from Kaggle fake job description prediction dataset.This dataset contains 18K job descriptions out of which about 800 are fake. The data consists of both textual information and meta-information about the jobs. Data set contains categorical features as well as some features containing text data.

#### Data Exploration : 

From the dataset, we can see which industry is offering more jobs, the chart below shows that information
![Industries Graph](https://github.com/Pranjal-Soni/fake_news_description_prediction/blob/master/images/top_20_industries.png)

#### Data Cleaning and Preprocessing :
The dataset contains numerical and text data both and there are null also in columns. Mostly numerical columns like telecommuting, has_company_logo, has_question have categorical data and text feature like employment_type, eductaion_requirment etc. So to deal with a categorical feature one-hot encoding is applied for the categorical columns. Preprocessing of text data are being done in several steps. The steps are as follow removing links from text data, lower all text data, remove stop words using nltk library and then check for most important words like which words have more occurrence and which words have less occurrence and according to that remove all non-important words. Finally, I deal with nan values by appending info not given with column name for each column so for categorical features they have one feature. To convert text data into numerical tfidf transformation method is used. Some visulisations after data preprocessing :
![Histograms](https://github.com/Pranjal-Soni/fake_news_description_prediction/blob/master/images/visualise_featues.png)

#### Handling Imbalanced Dataset:
The dataset is highly imbalanced because it contains 866 fraudulent data and 17014 non-fraudulent data. Fraudulent data is only 5% of the dataset. So if we apply any machine learning algorithm without handling imbalanced data. The model gives us a high accuracy. But the model is not correct or proper. So to handle this problem I used resampling method. In this method oversampling to is used to in training data. What is basically is done that to make 50% ratio of each class we took fraudulent data and resample it of the size of non-fradulent data. Thus each class have an equal ratio in the training data and no more imbalance is there.

#### Applying Different Machine Learning Models:
* Naive Bayes Classifier : Frist we train our model for Naive Baye's classifier. It is a good machine learning model with a probilistic approch. This good for sparse data and handling too many feature. Intinally we are getting auc_roc score, after hyperparameter tuning we are able to get 70% auc_roc score for our model.

* SVM Classifier : Support vector machine classifer is used to draw the decision boundry between fraudulent and non-fraudulent job discriptions. Initially we are getting 74% roc_auc score for test data after hyperparameter tuning we are able to get 85% roc_auc score for the test data.

* Random Forest Classifier : Random Forest classifier is widly used ensemble machine learning model. Initially random forest is giving 74% roc_auc score and after hyperparameter tuning we get 78% roc_auc score.

#### Result :
Hyperparameter tuning hepls a lot to improve the accuracy of different models. Naive bayes model has better accuray for fraudulent jobs and it classify more fraudulent jobs correctly, but it also classify a lot of non-fraudulent jobs as fraudulent which is major drawback of naive bayes model for this project. The SVM model is very balanced model and it gives best roc_auc score. It is able to correctly classify most of the fraudulent jobs and non fraudulent and the random forest model is good for classifying the non-fradulent jobs. So we can say that the svm model is the best fit for this problem.


