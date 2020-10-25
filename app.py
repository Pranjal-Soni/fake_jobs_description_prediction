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






def edu_list(param):
    arr = [0,0,0,0,0,0,0,0,0,0,0,0,0]
    if param == 'Associate Degree':
        arr[0] = 1
        return arr
    elif param == "Bachelor's Degree":
        arr[1] = 1
        return arr
    elif param == "Certification":
        arr[2] = 1
        return arr
    elif param == "Doctorate":
        arr[3] = 1
        return arr
    elif param == "High School or equivalent":
        arr[4] = 1
        return arr
    elif param == "Master's Degree":
        arr[5] = 1
        return arr
    elif param == "Professional":
        arr[6] = 1
        return arr
    elif param == "Some College Coursework Completed":
        arr[7] = 1
        return arr
    elif param == "Some High School Coursework":
        arr[8] = 1
        return arr
    elif param == "Unspecified":
        arr[9] = 1
        return arr
    elif param == "Vocational":
        arr[10] = 1
        return arr
    elif param == "Vocational - Degree":
        arr[11] = 1
        return arr
    elif param == "Vocational - HS Diploma":
        arr[12] = 1
        return arr
    else:
        return arr

def employment_list(param):
    arr = [0,0,0,0,0,0]
    if param == "Contract":
        arr[0] = 1
        return arr
    elif param == "Full-time":
        arr[1] = 1
        return arr
    elif param == "Other":
        arr[2] = 1
        return arr
    elif param == "Part-time":
        arr[3] = 1
        return arr
    elif param == "Temporary":
        arr[4] = 1
        return arr
    elif param == "no_info_about_employment":
        arr[5] = 1
        return arr
    else:
        return arr

def experience_list(param):
    arr = [0,0,0,0,0,0,0,0]
    if param == "Associate":
        arr[0] = 1
        return arr
    elif param == "Director":
        arr[1] = 1
        return arr
    elif param == "Entry level":
        arr[2] = 1
        return arr
    elif param == "Executive":
        arr[3] = 1
        return arr
    elif param == "Internship":
        arr[4] = 1
        return arr
    elif param == "Mid-Senior level":
        arr[5] = 1
        return arr
    elif param == "Not Applicable":
        arr[6] = 1
        return arr
    elif param == "experience_not_asked":
        arr[7] = 1
        return arr
    else:
        return arr

def function_list(param):
    arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    if param == "Accounting/Auditing":
        arr[0] = 1
        return arr
    elif param == "Administrative":
        arr[1] = 1
        return arr
    elif param == "Advertising":
        arr[2] = 1
        return arr
    elif param == "Art/Creative":
        arr[3] = 1
        return arr
    elif param == "Business Analyst":
        arr[4] = 1
        return arr
    elif param == "Business Development":
        arr[5] = 1
        return arr
    elif param == "Consulting":
        arr[6] = 1
        return arr
    elif param == "Customer Service":
        arr[7] = 1
        return arr
    elif param == "Data Analyst":
        arr[8] = 1
        return arr
    elif param == "Design":
        arr[9] = 1
        return arr
    elif param == "Distribution":
        arr[10] = 1
        return arr
    elif param == "Education":
        arr[11] = 1
        return arr
    elif param == "Engineering":
        arr[12] = 1
        return arr
    elif param == "Finance":
        arr[13] = 1
        return arr
    elif param == "Financial Analyst":
        arr[14] = 1
        return arr
    elif param == "General Business":
        arr[15] = 1
        return arr
    elif param == "Health Care Provider":
        arr[16] = 1
        return arr
    elif param == "Human Resources":
        arr[17] = 1
        return arr
    elif param == "Information Technology":
        arr[18] = 1
        return arr
    elif param == "Legal":
        arr[19] = 1
        return arr
    elif param == "Management":
        arr[20] = 1
        return arr
    elif param == "Manufacturing":
        arr[21] = 1
        return arr
    elif param == "Marketing":
        arr[22] = 1
        return arr
    elif param == "Other":
        arr[23] = 1
        return arr
    elif param == "Product Management":
        arr[24] = 1
        return arr
    elif param == "Production":
        arr[25] = 1
        return arr
    elif param == "Project Management":
        arr[26] = 1
        return arr
    elif param == "Public Relations":
        arr[27] = 1
        return arr
    elif param == "Purchasing":
        arr[28] = 1
        return arr
    elif param == "Quality Assurance":
        arr[29] = 1
        return arr
    elif param == "Research":
        arr[30] = 1
        return arr
    elif param == "Sales":
        arr[31] = 1
        return arr
    elif param == "Science":
        arr[32] = 1
        return arr
    elif param == "Strategy/Planning":
        arr[33] = 1
        return arr
    elif param == "Supply Chain":
        arr[34] = 1
        return arr
    elif param == "Training":
        arr[35] = 1
        return arr
    elif param == "Writing/Editing":
        arr[36] = 1
        return arr
    elif param == "function_not_given":
        arr[37] = 1
        return arr
    else:
        return arr


def industry_list(param):
    arr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0]
    
    if param == "Accounting":
        arr[0] = 1
        return arr
    elif param == "Airlines/Aviation":
        arr[1] = 1
        return arr
    elif param == "Alternative Dispute Resolution":
        arr[2] = 1
        return arr
    elif param == "Animation":
        arr[3] = 1
        return arr
    elif param == "Apparel & Fashion":
        arr[4] = 1
        return arr
    elif param == "Architecture & Planning":
        arr[5] = 1
        return arr
    elif param == "Automotive":
        arr[6] = 1
        return arr
    elif param == "Aviation & Aerospace":
        arr[7] = 1
        return arr
    elif param == "Banking":
        arr[8] = 1
        return arr
    elif param == "Biotechnology":
        arr[9] = 1
        return arr
    elif param == "Broadcast Media":
        arr[10] = 1
        return arr
    elif param == "Building Materials":
        arr[11] = 1
        return arr
    elif param == "Business Supplies and Equipment":
        arr[12] = 1
        return arr
    elif param == "Capital Markets":
        arr[13] = 1
        return arr
    elif param == "Chemicals":
        arr[14] = 1
        return arr
    elif param == "Civic & Social Organization":
        arr[15] = 1
        return arr
    elif param == "Civil Engineering":
        arr[16] = 1
        return arr
    elif param == "Commercial Real Estate":
        arr[17] = 1
        return arr
    elif param == "Computer & Network Security":
        arr[18] = 1
        return arr
    elif param == "Computer Games":
        arr[19] = 1
        return arr
    elif param == "Computer Hardware":
        arr[20] = 1
        return arr
    elif param == "Computer Networking":
        arr[21] = 1
        return arr
    elif param == "Computer Software":
        arr[22] = 1
        return arr
    elif param == "Construction":
        arr[23] = 1
        return arr
    elif param == "Consumer Electronics":
        arr[24] = 1
        return arr
    elif param == "Consumer Goods":
        arr[25] = 1
        return arr
    elif param == "Consumer Services":
        arr[26] = 1
        return arr
    elif param == "Cosmetics":
        arr[27] = 1
        return arr
    elif param == "Defense & Space":
        arr[28] = 1
        return arr
    elif param == "Design":
        arr[29] = 1
        return arr
    elif param == "E-Learning":
        arr[30] = 1
        return arr
    elif param == "Education Management":
        arr[31] = 1
        return arr
    elif param == "Electrical/Electronic Manufacturing":
        arr[32] = 1
        return arr
    elif param == "Entertainment":
        arr[33] = 1
        return arr
    elif param == "Environmental Services":
        arr[34] = 1
        return arr
    elif param == "Events Services":
        arr[35] = 1
        return arr
    elif param == "Executive Office":
        arr[36] = 1
        return arr
    elif param == "Facilities Services":
        arr[37] = 1
        return arr
    elif param == "Farming":
        arr[38] = 1
        return arr
    elif param == "Financial Services":
        arr[39] = 1
        return arr
    elif param == "Fishery":
        arr[40] = 1
        return arr
    elif param == "Food & Beverages":
        arr[41] = 1
        return arr
    elif param == "Food Production":
        arr[42] = 1
        return arr
    elif param == "Fund-Raising":
        arr[43] = 1
        return arr
    elif param == "Furniture":
        arr[44] = 1
        return arr
    elif param == "Gambling & Casinos":
        arr[45] = 1
        return arr
    elif param == "Government Administration":
        arr[46] = 1
        return arr
    elif param == "Government Relations":
        arr[47] = 1
        return arr
    elif param == "Graphic Design":
        arr[48] = 1
        return arr
    elif param == "Health, Wellness and Fitness":
        arr[49] = 1
        return arr
    elif param == "Higher Education":
        arr[50] = 1
        return arr
    elif param == "Hospital & Health Care":
        arr[51] = 1
        return arr
    elif param == "Hospitality":
        arr[52] = 1
        return arr
    elif param == "Human Resources":
        arr[53] = 1
        return arr
    elif param == "Import and Export":
        arr[54] = 1
        return arr
    elif param == "Individual & Family Services":
        arr[55] = 1
        return arr
    elif param == "Industrial Automation":
        arr[56] = 1
        return arr
    elif param == "Information Services":
        arr[57] = 1
        return arr
    elif param == "Information Technology and Services":
        arr[58] = 1
        return arr
    elif param == "Insurance":
        arr[59] = 1
        return arr
    elif param == "International Trade and Development":
        arr[60] = 1
        return arr
    elif param == "Internet":
        arr[61] = 1
        return arr
    elif param == "Investment Banking":
        arr[62] = 1
        return arr
    elif param == "Investment Management":
        arr[63] = 1
        return arr
    elif param == "Law Enforcement":
        arr[64] = 1
        return arr
    elif param == "Law Practice":
        arr[65] = 1
        return arr
    elif param == "Legal Services":
        arr[66] = 1
        return arr
    elif param == "Leisure, Travel & Tourism":
        arr[67] = 1
        return arr
    elif param == "Libraries":
        arr[68] = 1
        return arr
    elif param == "Logistics and Supply Chain":
        arr[69] = 1
        return arr
    elif param == "Luxury Goods & Jewelry":
        arr[70] = 1
        return arr
    elif param == "Machinery":
        arr[71] = 1
        return arr
    elif param == "Management Consulting":
        arr[72] = 1
        return arr
    elif param == "Maritime":
        arr[73] = 1
        return arr
    elif param == "Market Research":
        arr[74] = 1
        return arr
    elif param == "Marketing and Advertising":
        arr[75] = 1
        return arr
    elif param == "Mechanical or Industrial Engineering":
        arr[76] = 1
        return arr
    elif param == "Media Production":
        arr[77] = 1
        return arr
    elif param == "Medical Devices":
        arr[78] = 1
        return arr
    elif param == "Medical Practice":
        arr[79] = 1
        return arr
    elif param == "Mental Health Care":
        arr[80] = 1
        return arr
    elif param == "Military":
        arr[81] = 1
        return arr
    elif param == "Mining & Metals":
        arr[82] = 1
        return arr
    elif param == "Motion Pictures and Film":
        arr[83] = 1
        return arr
    elif param == "Museums and Institutions":
        arr[84] = 1
        return arr
    elif param == "Music":
        arr[85] = 1
        return arr
    elif param == "Nanotechnology":
        arr[86] = 1
        return arr
    elif param == "Nonprofit Organization Management":
        arr[87] = 1
        return arr
    elif param == "Oil & Energy":
        arr[88] = 1
        return arr
    elif param == "Online Media":
        arr[89] = 1
        return arr
    elif param == "Outsourcing/Offshoring":
        arr[90] = 1
        return arr
    elif param == "Package/Freight Delivery":
        arr[91] = 1
        return arr
    elif param == "Packaging and Containers":
        arr[92] = 1
        return arr
    elif param == "Performing Arts":
        arr[93] = 1
        return arr
    elif param == "Pharmaceuticals":
        arr[94] = 1
        return arr
    elif param == "Philanthropy":
        arr[95] = 1
        return arr
    elif param == "Photography":
        arr[96] = 1
        return arr
    elif param == "Plastics":
        arr[97] = 1
        return arr
    elif param == "Primary/Secondary Education":
        arr[98] = 1
        return arr
    elif param == "Printing":
        arr[99] = 1
        return arr
    elif param == "Professional Training & Coaching":
        arr[100] = 1
        return arr
    elif param == "Program Development":
        arr[101] = 1
        return arr
    elif param == "Public Policy":
        arr[102] = 1
        return arr
    elif param == "Public Relations and Communications":
        arr[103] = 1
        return arr
    elif param == "Public Safety":
        arr[104] = 1
        return arr
    elif param == "Publishing":
        arr[105] = 1
        return arr
    elif param == "Ranching":
        arr[106] = 1
        return arr
    elif param == "Real Estate":
        arr[107] = 1
        return arr
    elif param == "Religious Institutions":
        arr[108] = 1
        return arr
    elif param == "Renewables & Environment":
        arr[109] = 1
        return arr
    elif param == "Research":
        arr[110] = 1
        return arr
    elif param == "Restaurants":
        arr[111] = 1
        return arr
    elif param == "Retail":
        arr[112] = 1
        return arr
    elif param == "Security and Investigations":
        arr[113] = 1
        return arr
    elif param == "Semiconductors":
        arr[114] = 1
        return arr
    elif param == "Shipbuilding":
        arr[115] = 1
        return arr
    elif param == "Sporting Goods":
        arr[116] = 1
        return arr
    elif param == "Sports":
        arr[117] = 1
        return arr
    elif param == "Staffing and Recruiting":
        arr[118] = 1
        return arr
    elif param == "Telecommunications":
        arr[119] = 1
        return arr
    elif param == "Textiles":
        arr[120] = 1
        return arr
    elif param == "Translation and Localization":
        arr[121] = 1
        return arr
    elif param == "Transportation/Trucking/Railroad":
        arr[122] = 1
        return arr
    elif param == "Utilities":
        arr[123] = 1
        return arr
    elif param == "Venture Capital & Private Equity":
        arr[124] = 1
        return arr
    elif param == "Veterinary":
        arr[125] = 1
        return arr
    elif param == "Warehousing":
        arr[126] = 1
        return arr
    elif param == "Wholesale":
        arr[127] = 1
        return arr
    elif param == "Wine and Spirits":
        arr[128] = 1
        return arr
    elif param == "Wireless":
        arr[129] = 1
        return arr
    elif param == "Writing and Editing":
        arr[130] = 1
        return arr
    elif param == "industry_not_given":
        arr[131] = 1
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