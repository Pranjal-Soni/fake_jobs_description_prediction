#importing important libraries
import re
import config
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


#remove links from the dataset
def remove_links(text):
    text  = re.sub(r'http://[\w|\S]+',' ',str(text))
    return text


#trasfrom all the words in lowercase
def lower_text(text):
    text = str(text)
    text = ' '.join(x.lower() for x in text.split())
    return text


#remove stop words from the text
stop = stopwords.words('english')
def remove_stopWords(text):
    text = str(text)
    text = ''.join( x for x in text if x.split() not in stop)
    return(text)

#remove the words which are not important
def remove_non_imp_words(series):
    most_freq = pd.Series(' '.join(series).split()).value_counts()[:30]
    less_freq = pd.Series(' '.join(series).split()).value_counts()[-30:]
    series = series.apply(lambda x: " ".join(x for x in x.split() if x not in most_freq))
    series = series.apply(lambda x: " ".join(x for x in x.split() if x not in less_freq))
    return series

#hadling none values
def remove_nan(text):
    if text == 'nan' or text == '':
        text = 'not given'
    return text

#apply all the necessory funtion to clean text data
def data_cleaning(feature):
    WNlemma = nltk.WordNetLemmatizer()
    feature = feature.apply(lambda x : remove_links(x))
    feature = feature.apply(lambda x : lower_text(x))
    feature = feature.apply(lambda x : remove_stopWords(x))
    feature = feature.str.replace('[^\w\s]',' 111')
    feature = remove_non_imp_words(feature)
    feature = feature.apply(lambda x: ' '.join([WNlemma.lemmatize(word) for word in x.split()]))
    feature = feature.apply(lambda x : remove_nan(x))
    return feature

if __name__=="__main__":
    data = pd.read_csv(config.MAIN_DATASET)
    target = data['fraudulent']
    #drop department and salary_range because it have about 60 persent null values and removing irrelavent data from the dataset
    data.drop(['job_id','salary_range','department','benefits'],axis=1,inplace=True)
    #filling missing values of some columns manaully
    data['required_education'].fillna('no_info_about_education',inplace = True)
    data['employment_type'].fillna('no_info_about_employment',inplace = True)
    data['required_experience'].fillna('experience_not_asked',inplace = True)
    data['industry'].fillna('industry_not_given',inplace = True)
    data['function'].fillna('function_not_given',inplace = True)
    
    #dealing with company profile feature
    for i in range(len(data.company_profile)):
        if data.company_profile[i]=='NaN':
            data.company_profile[i] = 'company_profile_not_given'
        else:
            data.company_profile[i] = 'company_profile_given'
    
    #dealing catogorical data
    cat_cols = ['employment_type','required_experience','required_education','industry','function','company_profile']
    for c in cat_cols:
        encoded = pd.get_dummies(data[c])
        data = pd.concat([data,encoded],axis = 1 )
    cat_cols = ['employment_type','required_experience','required_education','industry','function','title','location','company_profile']
    data.drop(cat_cols,axis=1,inplace=True)
    
    #dealing with text data
    description = data['description']+ ' ' +  data['requirements']
    description = data_cleaning(description)

    tfidf = TfidfVectorizer( min_df = 0.05, ngram_range=(1,3))
    tfidf_features = tfidf.fit_transform(description) 
    tfidf_vect_df = pd.DataFrame(tfidf_features.todense(), columns = tfidf.get_feature_names())
    data = pd.concat([data, tfidf_vect_df], axis = 1)
    data.drop(['description','requirements'],axis = 1 , inplace = True)
    
    #suffle the dataset split in train and test
    data = data.sample(frac=1).reset_index(drop=True)
    
    #selcting the 80% data as train and 20% data for test
    train = data.loc[:len(data)*0.8]
    test = data.loc[len(data)*0.8:]
    
    print(f'Train data have {len(train[train.fraudulent==1])} jobs.')
    print(f'Test data have {len(test[test.fraudulent==1])} jobs.')
    train.to_csv('../inputs/train.csv',index=False)
    test.to_csv('../inputs/test.csv',index=False)