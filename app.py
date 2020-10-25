import numpy as np
import pandas as pd
import pickle
from flask import Flask,request,jsonify,render_template
import texthero
import nltk
from nltk import stem
from nltk import tokenize
from nltk.stem.snowball import SnowballStemmer
import texthero as hero
import re

app = Flask(__name__)
tfidf = pickle.load(open('./inputs/tf_idf.pkl','rb'))
model = pickle.load(open('./models/final_model.pkl','rb'))
column_entities = pickle.load(open('./inputs/column_entities.pkl','rb'))


w_tokenizer = tokenize.WhitespaceTokenizer()
stemmer = stem.SnowballStemmer("english")
def stemming(text):
    '''
    input  : text 
    output :  lemmazitzed text
    '''
    return ' '.join([stemmer.stem(word) for word in w_tokenizer.tokenize(text)])

def remove_long_texts(text):
    words = [word for word in text.split() if len(word)<21]
    return ' '.join(words)

def remove_digits(text):
    return re.sub(r'[\d\|]', '', text)

def clean(text):
    s = pd.Series([text])
    text = s.pipe(hero.clean)[0]
    text = remove_long_texts(text)
    text = remove_digits(text)
    text = stemming(text)
    return tfidf.transform([text]).toarray()[0]


def employment_list(param):
    employments = column_entities[0]
    arr = np.zeros((1,len(employments)),dtype=int).tolist()[0]
    index = employments.index(param)
    if index is not None:
        arr[index] = 1
        return arr
    else:
        return arr

def experience_list(param):
    arr = [0,0,0,0,0,0,0,0]
    experiences = column_entities[1]
    arr = np.zeros((1,len(experiences)),dtype=int).tolist()[0]
    index = experiences.index(param)
    if index is not None:
        arr[index] = 1
        return arr
    else:
        return arr

def edu_list(param):
    educations = column_entities[2]
    arr = np.zeros((1,len(educations)),dtype=int).tolist()[0]
    index = educations.index(param)
    if index is not None:
        arr[index] = 1
        return arr
    else:
        return arr

def industry_list(param):
    industries = column_entities[3]
    arr = np.zeros((1,len(industries)),dtype=int).tolist()[0]
    index = industries.index(param)
    if index is not None:
        arr[index] = 1
        return arr
    else:
        return arr

def function_list(param):
    functions = column_entities[4]
    arr = np.zeros((1,len(functions)),dtype=int).tolist()[0]
    index = functions.index(param)
    if index is not None:
        arr[index] = 1
        return arr
    else:
        return arr

def preaper_data(clean_desc,required_experience,employment_type,has_questions
            ,company_logo,telecommuting,required_education,industry,function):
    
    result = []
    result = result + [int(telecommuting)]
    result = result + [int(company_logo)]
    result = result + [int(has_questions)]
    result = result + edu_list(required_education)
    result = result + employment_list(employment_type)
    result = result + experience_list(required_experience)
    result = result +  industry_list(industry)
    result = result + function_list(function)
    result = result + list(clean_desc)
    return pd.Series(result)


@app.route('/')
def home():
    return render_template('index.html',prediction=None)

@app.route('/predict',methods = ['POST'])
def predict():
    if request.method == 'POST':
        title = request.form['title']
        description = request.form['description']
        requirment = request.form['requirment']
        required_experience = request.form['required_experience']
        employment_type = request.form['employment_type']
        has_questions = request.form['has_questions']
        company_logo = request.form['company_logo']
        telecommuting = request.form['telecommuting']
        required_education = request.form['required_education']
        industry = request.form['industry']
        function = request.form['function']
        desc = title + description+requirment
        clean_desc = clean(desc)

        if len(set(clean_desc)) == 1:
            predict = [1]
        else:
            data = preaper_data(clean_desc,required_experience,employment_type,has_questions
                ,company_logo,telecommuting,required_education,industry,function)
            predict = model.predict(data.values.reshape(1,-1))
        return render_template('result.html',prediction = predict)

if __name__=="__main__":
    app.run(debug=True)