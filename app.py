"""
This script runs the application using a development server.
It contains the definition of routes and views for the application.
"""
import os
from flask import Flask, render_template, make_response, request, redirect, url_for
from flask import send_from_directory
from flask_restful import Api, Resource
from gensim.models import Word2Vec

# Neural Network headers
from pandas import read_csv
from sklearn.metrics import confusion_matrix,multilabel_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Embedding, GRU, SimpleRNN
from keras.utils import to_categorical
import pandas as pd
import csv
import os
import io
import requests
from keras import backend as K
import matplotlib.pyplot as plt
import xgboost

#RNN libraries
###############################################################################
import keras
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import nltk 
from nltk.corpus import stopwords
import re
import tensorflow as tf
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer,PorterStemmer
from keras.models import load_model
from keras.models import model_from_json
from keras_lookahead import Lookahead
#!pip install keras-lookahead

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
###############################################################################

#Decision tree headers
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import nltk 
from nltk.corpus import stopwords
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer,PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer


#KNN classifier headers
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline


#SVM classifier headers
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
#!pip install gensim
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
import gensim
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import TaggedDocument
import re
import seaborn as sns
import matplotlib.pyplot as plt
#nltk.download('punkt')
#LogisticRegression libraries
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
#!pip install gensim
#!pip install wget
from gensim.models import Doc2Vec
from gensim.models import Word2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
from sklearn import svm 
import gensim
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import TaggedDocument
import re
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')
from sklearn.metrics import accuracy_score
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
#!pip install gensim
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.model_selection import train_test_split
import gensim
from sklearn.linear_model import LogisticRegression
from gensim.models.doc2vec import TaggedDocument
import re
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import model_selection, naive_bayes, svm 
from sklearn.metrics import accuracy_score

incident = ""
xgb_prb = ""
xgb_pred = ""

app = Flask(__name__,static_url_path='',static_folder='static')
app.config['USE_FAVICON'] = os.path.exists(os.path.join(app.static_folder, "favicon.ico"))
api = Api(app)

# Make the WSGI interface available at the top level so wfastcgi can get it.
wsgi_app = app.wsgi_app

class NLP(Resource):
      
    def get(self):
        
        headers = {'Content-Type': 'text/html'}
        return make_response(render_template("index3.html"), 201,headers)

###############  Text preprocessing functions ############
##### LR funcs start #######
def lr_clean_text(text):
    text = text.lower() # lowercase text
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    numerical_symbols = re.compile('0-90-9a-z')
    STOPWORDS = set(stopwords.words('english'))
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    text = numerical_symbols.sub('', text)
    text = text.replace('x', '')
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
    return text

def lr_getLemmText(text):
    tokens=word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens=[lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)


def lr_getStemmText(text):
    tokens=word_tokenize(text)
    ps = PorterStemmer()
    tokens=[ps.stem(word) for word in tokens]
    return ' '.join(tokens)


##### LR funcs end #######

##### SVM funcs start #######
def svm_clean_text(text):
    text = text.lower() # lowercase text
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    numerical_symbols = re.compile('0-90-9a-z')
    STOPWORDS = set(stopwords.words('english'))
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    text = numerical_symbols.sub('', text)
    text = text.replace('x', '')
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
    return text

def svm_getLemmText(text):
    tokens=word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens=[lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)


def svm_getStemmText(text):
    tokens=word_tokenize(text)
    ps = PorterStemmer()
    tokens=[ps.stem(word) for word in tokens]
    return ' '.join(tokens)


##### SVM funcs end #######


##### RF funcs #######
def rf_preprocess(my_text):
  df = pd.DataFrame({'EventDescription':[my_text]})
  df.rename(columns = {'EventDescription':'EventDescription_parsed'}, inplace = True)

  df['EventDescription_parsed'].apply(lambda x: len(x.split(' '))).sum()

  #Remove punctuations
  punctuation_signs = list("?:!.,;")

  for punct_sign in punctuation_signs:
      df['EventDescription_parsed'].str.replace(punct_sign, '')
  df['EventDescription_parsed'] = df['EventDescription_parsed'].apply(rf_cleanText)
  return df.loc[0].values # processes text

def rf_cleanText(text):
    punctuation_signs = list("?:!.,;")
    text = text.lower()
    for punct_sign in punctuation_signs:
        text = text.replace(punct_sign, '')
    return text


##### RF funcs #######

##### DT funcs #######
def dt_preprocess(my_text):
  pima = pd.DataFrame({'EventDescription':[my_text]})
  df=pd.DataFrame()
  df = df.reset_index(drop=True)
  pima['EventDescription'] = pima['EventDescription'].apply(dt_clean_text)
  pima['EventDescription'] = list(map(dt_getLemmText,pima['EventDescription']))
  pima['EventDescription'] = list(map(dt_getStemmText,pima['EventDescription']))
  df = pima[['EventDescription']]
  df = df[pd.notnull(df['EventDescription'])]
  return pima['EventDescription']

def dt_clean_text(text):
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    numerical_symbols = re.compile('0-90-9a-z')
    STOPWORDS = set(stopwords.words('english'))
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    text = numerical_symbols.sub('', text)
    text = text.replace('x', '')
    #text = re.sub(r'\W+', '', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
    return text

def dt_getLemmText(text):
    tokens=word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens=[lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

def dt_getStemmText(text):
    tokens=word_tokenize(text)
    ps = PorterStemmer()
    tokens=[ps.stem(word) for word in tokens]
    return ' '.join(tokens)

##### DT funcs #######

##### RNN funcs #######

def rnn_clean_text(text):
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    STOPWORDS = set(stopwords.words('english'))
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    text = text.replace('x', '')
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
    return text

def rnn_getLemmText(text):
    tokens=word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    tokens=[lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)
       
def rnn_getStemmText(text):
    tokens=word_tokenize(text)
    ps = PorterStemmer()
    tokens=[ps.stem(word) for word in tokens]
    return ' '.join(tokens)


def rnn_preprocess(desc):
    d = {'words' : [desc]}
    df = pd.DataFrame(data = d)
    df['words'] = df['words'].apply(rnn_clean_text)
    df['words'] = list(map(rnn_getLemmText,df['words']))
    df['words'] = list(map(rnn_getStemmText,df['words']))
    return df['words']
######### RNN ###########################


######### KNN ###########################
def knn_cleanText(text):
    punctuation_signs = list("?:!.,;")
    text = text.lower()
    for punct_sign in punctuation_signs:
        text = text.replace(punct_sign, '')
    return text
######### KNN ###########################



###############  Text preprocessing functions end ############


########### modular funcs for each classifier ###############

def xgb_predictor(incident):
    xgb_model = pickle.load(open("Models/xgb/model.sav", 'rb'))
    cv = pickle.load(open("Models/xgb/count_vector.pickel", "rb"))
    le = pickle.load(open("Models/xgb/label_encoder.pickel", "rb"))
    desc_new = cv.transform([incident])
    prob = xgb_model.predict_proba(desc_new)
    pred = np.argmax(prob)
    xgb_pred = le.inverse_transform([pred])[0]
    xgb_prob = str(np.round(np.max(prob) * 100,2)) + "%"
    return xgb_pred, xgb_prob

def lr_predictor(incident):
    logreg = pickle.load(open("Models/lr/logreg.sav", 'rb'))
    vect = pickle.load(open("Models/lr/vect.pickel", "rb"))
    le = pickle.load(open("Models/lr/le.pickel", "rb"))

    d = {'words' : [incident]}
    df = pd.DataFrame(data = d)
    df['words'] = df['words'].apply(lr_clean_text)
    df['words'] = list(map(lr_getLemmText,df['words']))
    df['words'] = list(map(lr_getStemmText,df['words']))
    transformed = vect.transform(df['words'])
    answer = logreg.predict(transformed)
    lr_pred = le.inverse_transform(answer)[0]
    prob = logreg.predict_proba(transformed)
    lr_prob = str(np.round(np.max(prob) * 100,2)) + "%"
    return lr_pred, lr_prob

def svm_predictor(incident):
    svm_loaded = pickle.load(open("Models/svm/svm2.sav", 'rb'))
    vect = pickle.load(open("Models/svm/vect.pickel", "rb"))
    le = pickle.load(open("Models/svm/le.pickel", "rb"))
    d = {'words' : [incident]}
    df = pd.DataFrame(data = d)
    df['words'] = df['words'].apply(svm_clean_text)
    df['words'] = list(map(svm_getLemmText,df['words']))
    df['words'] = list(map(svm_getStemmText,df['words']))
    transformed = vect.transform(df['words'])
    # TODO this is not suppported yet although it works in google colab
    # TRansformed has 3 dimesions here , whereas it has 2 dimensions in google collab
    answer = svm_loaded.predict(transformed)
    pred = le.inverse_transform(answer)[0]
    prob = svm_loaded.predict_proba(transformed)
    svm_prob = str(np.round(np.max(prob) * 100,2)) + "%"
    return pred, svm_prob

def rf_predictor(incident):
    rf_model = pickle.load(open("Models/rf/randomforest_model.sav", 'rb'))
    cv = pickle.load(open("Models/rf/randomforest_vector.pickel", "rb"))
    tf = pickle.load(open("Models/rf/randomforest_transformer.pickel", "rb"))
    vect_rf = cv.transform(rf_preprocess(incident))
    trans_rf = tf.transform(vect_rf)
    rf_pred = rf_model.predict(trans_rf)[0]
    prob = rf_model.predict_proba(trans_rf)
    rf_prob = str(np.round(np.max(prob) * 100,2)) + "%"
    return rf_pred, rf_prob

def dt_predictor(incident):
    dt_model = pickle.load(open("Models/dt/decisiontree.sav", 'rb'))
    cv = pickle.load(open("Models/dt/decisiontree_vector.pickel", "rb"))
    le = pickle.load(open("Models/dt/decisiontree_labelenc.pickel", "rb"))
    trans_dt = cv.transform(dt_preprocess(incident))
    my_pred = dt_model.predict(trans_dt)
    dt_pred = le.inverse_transform(my_pred)[0]
    prob = dt_model.predict_proba(trans_dt)
    dt_prob = str(np.round(np.max(prob) * 100,2)) + "%"
    return dt_pred, dt_prob

def knn_predictor(incident):
    knn_model = pickle.load(open("Models/knn/saved_model_knn.sav", 'rb'))
    knn_cv = pickle.load(open("Models/knn/saved_vectorizer.pickle", 'rb'))
    knn_tf = pickle.load(open("Models/knn/saved_tfidf_trans.pickle", 'rb'))

    docs_new = knn_cleanText(incident)
    docs_new = [docs_new]
    X_new_counts = knn_cv.transform(docs_new)
    X_new_tfidf = knn_tf.transform(X_new_counts)
    predicted = knn_model.predict(X_new_tfidf)[0]
    prob = knn_model.predict_proba(X_new_tfidf)
    knn_prob = str(np.round(np.max(prob) * 100,2)) + "%"
    return predicted, knn_prob 

  
def rnn_predictor(incident):
    json_file = open('Models/rnn/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("Models/rnn/model.h5")
    tok = pickle.load(open("Models/rnn/tok.pickel", "rb"))
    le = pickle.load(open("Models/rnn/le.pickel", "rb"))

    optimiser = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
    lk = Lookahead(optimiser, sync_period=5, slow_step=0.5, name = 'Lookahead')
    loaded_model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=lk.optimizer,
                  metrics=['accuracy'])
    
    sequences = tok.texts_to_sequences(rnn_preprocess(incident))
    sequence_matrix = sequence.pad_sequences(sequences,maxlen=150)
    y_pred = loaded_model.predict_classes(sequence_matrix)
    category = le.inverse_transform(y_pred)[0]
    rnn_prob = loaded_model.predict_proba(sequence_matrix)
    rnn_prob = str(np.round(np.max(rnn_prob) * 100,2)) + "%"
    return category,rnn_prob
   


########### modular funcs for each classifier ###############

######################### Classifier/predictions ############################

@app.route('/predict', methods= ['POST'])
def predict_one_by_one():
    xgb_prob = ""
    xgb_pred = ""
    lr_prob = ""
    lr_pred = ""
    svm_prob = ""
    svm_pred = ""
    rf_prob = ""
    rf_pred = ""
    dt_prob = ""
    dt_pred = ""
    rnn_pred= ""
    rnn_prob= ""
    knn_pred= ""
    knn_prob= ""

    classifier_name = str(request.form.get('classifier'))
    incident = str(request.form.get('event'))
    if(classifier_name == "xgb"):
        xgb_pred ,xgb_prob = xgb_predictor(incident)
    if(classifier_name == "lr"):
        lr_pred ,lr_prob = lr_predictor(incident)
    if(classifier_name == "svm"):
        svm_pred ,svm_prob = svm_predictor(incident)
    if (classifier_name == "rf"):
        rf_pred ,rf_prob = rf_predictor(incident)
    if (classifier_name == "dt"):
        dt_pred ,dt_prob = dt_predictor(incident)
    if(classifier_name == "rnn"):
        rnn_pred ,rnn_prob = rnn_predictor(incident)
    if(classifier_name == "knn"):
        knn_pred ,knn_prob = knn_predictor(incident)

    return (render_template('index3.html', scroll='table', incident = incident, 
                    xgb_pred = xgb_pred, xgb_prob = xgb_prob, 
                    lr_pred = lr_pred, lr_prob = lr_prob,
                    svm_pred = svm_pred, svm_prob = svm_prob,
                    rf_pred = rf_pred, rf_prob = rf_prob,
                    rnn_pred = rnn_pred, rnn_prob = rnn_prob,
                    knn_pred = knn_pred, knn_prob = knn_prob,
                    dt_pred = dt_pred, dt_prob = dt_prob
                    ))

@app.route('/predictall', methods= ['POST'])
def predict(): # predicts all available classifiers at the same time
    xgb_prob = ""
    xgb_pred = ""
    lr_prob = ""
    lr_pred = ""
    svm_prob = ""
    svm_pred = ""
    rf_prob = ""
    rf_pred = ""
    dt_prob = ""
    dt_pred = ""
    rnn_prob = ""
    rnn_pred = ""
    knn_pred= ""
    knn_prob= ""

    # init
    classifier_name = str(request.form.get('classifier'))
    incident = str(request.form.get('event'))

    # XGB
    xgb_pred ,xgb_prob = xgb_predictor(incident)
    # LR
    lr_pred ,lr_prob = lr_predictor(incident)
    # SVM
    svm_pred ,svm_prob = svm_predictor(incident)
    # Random Forest
    rf_pred ,rf_prob = rf_predictor(incident)
    # Decision Tree
    dt_pred ,dt_prob = dt_predictor(incident)
    #RNN
    rnn_pred ,rnn_prob = rnn_predictor(incident)
    # KNN
    knn_pred ,knn_prob = knn_predictor(incident)

    return (render_template('index3.html', scroll='table', incident = incident, 
                    xgb_pred = xgb_pred, xgb_prob = xgb_prob, 
                    lr_pred = lr_pred, lr_prob = lr_prob,
                    svm_pred = svm_pred, svm_prob = svm_prob,
                    rf_pred = rf_pred, rf_prob = rf_prob,
                    rnn_pred = rnn_pred, rnn_prob = rnn_prob,
                    knn_pred = knn_pred, knn_prob = knn_prob,
                    dt_pred = dt_pred, dt_prob = dt_prob
                    ))

###########Write to CSV###############

from csv import writer
@app.route('/add_record', methods= ['POST'])
def add_record():
	message = ""
    # Open file in append mode
	with open("new_incidents.csv", 'a+', newline='') as write_obj:
		elem = []
		elem.append(str(request.form.get('action')))
		elem.append(str(request.form.get('address')))
		elem.append(str(request.form.get('Assetlbl')))
		elem.append(str(request.form.get('causecmnty')))
		elem.append(str(request.form.get('causeenv')))
		elem.append(str(request.form.get('eventDesc')))
		elem.append(str(request.form.get('causepre')))
		elem.append(str(request.form.get('techcause')))
		elem.append(str(request.form.get('causeworkp')))
		elem.append(str(request.form.get('cnttype')))
		elem.append(str(request.form.get('injtype')))
		
		# Create a writer object from csv module
		csv_writer = writer(write_obj)
		# Add contents of list as last row in the csv file
		csv_writer.writerow(elem)

	return (render_template('index3.html', scroll="AddIncident", message="Incident added successfully!"))
############# Model classes ####################
class XGBoost(Resource):

	def get(self):
		cols = ['EventDescription', 'Category']
		df = pd.read_csv('/content/cleaned_incidents1.csv', usecols=cols)
		df = df.dropna(subset=['Category'])
		
		# Creating stopwords list
		nltk.download('stopwords')

		stopwords = nltk.corpus.stopwords.words('english')
		custom = ["reported", "found", "caused", "incident", "crew", "injuries", "location", "arrival", "grassfire", "fire", "ground", "pole"]
		for word in custom:
		  stopwords.append(word)
		
		# Splitting of data in test and train
		x_train, x_test, y_train, y_test = train_test_split(df['EventDescription'], df['Category'], 
                                                    test_size=0.3)

		# TF-IDF Vectorizer
		vectorizer = TfidfVectorizer(stop_words=stopwords, analyzer='word')
		tfidf = vectorizer.fit(df['EventDescription'])
		
		# Transforming x_train and x_test using TF-IDF
		x_train_tfidf = tfidf.transform(x_train)
		x_test_tfidf = tfidf.transform(x_test)
		
		# Creating XGBoost Classifier using TF-IDF

		model = XGBClassifier(learing_rate=0.001, verbosity = 2)
		model.fit(x_train_tfidf, y_train)
		
		# Model evaluation
		pred = model.predict(x_test_tfidf)

		accuracy = accuracy_score(y_test, pred)
		print("Accuracy using TF-IDF is: {0:.2f}%".format(accuracy * 100.0))
		
		# Transforming x_train and x_test using Count Vectorizer

		cv = CountVectorizer(stop_words=stopwords, analyzer='word')
		cv.fit(df['EventDescription'])

		x_train_count = cv.transform(x_train)
		x_test_count = cv.transform(x_test)
		
		# Creating XGBoost Classifier using Count Vectorizer
		model1 = XGBClassifier(learing_rate=0.001)
		model1.fit(x_train_count, y_train)
		
		# Model evaluation
		pred = model1.predict(x_test_count)

		accuracy = accuracy_score(y_test, pred)
		print("Accuracy using Count Vectorizer is: {0:.2f}%".format(accuracy * 100.0))

################################ KNN
class KNN(Resource):

    def post(self):
        df = pd.read_csv('cleaned_incidents1.csv')
        df = df[['EventDescription','Category','CauseEnvironment']]
        df = df[pd.notnull(df['EventDescription'])]
        df = df[pd.notnull(df['CauseEnvironment'])]
        df.rename(columns = {'EventDescription':'EventDescription_parsed'}, inplace = True)

        new = df["CauseEnvironment"].str.split(";", expand = True)
        print(new)

        df1 = df
        df1 = df1.append(new, ignore_index = False)

        df1 = pd.DataFrame([list(x) for x in df.pop('CauseEnvironment')], index=df.index).add_prefix('g')

        df.dropna(subset= ["Category"], inplace = True)

        df.index = range(1506)
        df['EventDescription_parsed'].apply(lambda x: len(x.split(' ')))

        #Remove punctuations
        punctuation_signs = list("?:!.,;")

        for punct_sign in punctuation_signs:
            df['EventDescription_parsed'].str.replace(punct_sign, '')
    
        df.loc[1]['EventDescription_parsed']

        def cleanText(text):
            punctuation_signs = list("?:!.,;")               
            text = text.lower()
            for punct_sign in punctuation_signs:
                text = text.replace(punct_sign, '')        
            return text
        df['EventDescription_parsed'] = df['EventDescription_parsed'].apply(cleanText)

        #Splitting into train/test

        from sklearn.model_selection import train_test_split
        train, test = train_test_split(df, test_size=0.3, random_state = 42)

        # Builds a dictionary of features and transforms documents to feature
        # vectors and convert our text documents to a
# matrix of token counts (CountVectorizer)
        count_vect = CountVectorizer()
        X_train_counts = count_vect.fit_transform(train.EventDescription_parsed)

# transform a count matrix to a normalized tf-idf representation (tf-idf
# transformer)
        tfidf_transformer = TfidfTransformer()
        X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

        #Building the model

        knn = KNeighborsClassifier(n_neighbors=15)

# training our classifier ; train_data.target will be having numbers assigned
# for each category in train data
        clf = knn.fit(X_train_tfidf, train.Category)

# Input Data to predict their classes of the given categories
        docs_new = ['there was a trunk.']
# building up feature vector of our input
        X_new_counts = count_vect.transform(docs_new)
# We call transform instead of fit_transform because it's already been fit
        X_new_tfidf = tfidf_transformer.transform(X_new_counts)


        # predicting the category of our input text: Will give out number for
        # category
        predicted = clf.predict(X_new_tfidf)
# predicting the category of our input text: Will give out number for category
        predicted = clf.predict(X_new_tfidf)


        # We can use Pipeline to add vectorizer -> transformer -> classifier
        # all in a one compound classifier
        text_clf = Pipeline([('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', knn),])
# Fitting our train data to the pipeline
        text_clf.fit(train.EventDescription_parsed, train.Category)

        docs_test = test.EventDescription_parsed
# Predicting our test data
        predicted = text_clf.predict(docs_test)
        print('We got an accuracy of',np.mean(predicted == test.Category) * 100, '% over the test data.')

######################## SVM
class SVM(Resource):

    def post(self):
        df = pd.read_csv(r'C:\Users\Admin\Documents\d2i---deakin-energy_2020t2\Datasets\ESV Data\cleaned_incidents1.csv')
        df = df[['EventDescription','Category']]
        df = df[pd.notnull(df['EventDescription'])]
        df.rename(columns = {'EventDescription':'EventDescription_parsed'}, inplace = True)

        df.dropna(subset= ["Category"], inplace = True)

        df.index = range(6489)
        df['EventDescription_parsed'].apply(lambda x: len(x.split(' ')))

        cnt_pro = df['Category'].value_counts()

        plt.figure(figsize=(12,4))
        sns.barplot(cnt_pro.index, cnt_pro.values, alpha=0.8)
        plt.ylabel('Number of Incidents', fontsize=12)
        plt.xlabel('Category', fontsize=12)
        plt.xticks(rotation=90)
        plt.show()

        #Print a particular incident
        def print_incident(index):
            example = df[df.index == index][['EventDescription_parsed', 'Category']].values[0]    
            if len(example) > 0:
                print(example[0])       
                print('Category:', example[1])


        #Remove punctuations
        punctuation_signs = list("?:!.,;")

        for punct_sign in punctuation_signs:
            df['EventDescription_parsed'].str.replace(punct_sign, '')
        df.loc[1]['EventDescription_parsed']

        def cleanText(text):

            punctuation_signs = list("?:!.,;")
            text = text.lower()
            for punct_sign in punctuation_signs:
                text = text.replace(punct_sign, '')
            return text

        df['EventDescription_parsed'] = df['EventDescription_parsed'].apply(cleanText)
        
        #Train|Test split
        train, test = train_test_split(df, test_size=0.3, random_state = 42)


        #Feeding GoogleNewsVector

        from gensim.models import Word2Vec
        wv = gensim.models.KeyedVectors.load_word2vec_format(r"C:\Users\Admin\Documents\d2i---deakin-energy_2020t2\Data Analysis\Deliverable 3\GoogleNews-vectors-negative300.bin.gz", binary=True)
        wv.init_sims(replace=True)

        def word_averaging(wv, words):
            all_words, mean = set(), []
    
    
        for word in words:
            if isinstance(word, np.ndarray):
                mean.append(word)
            
            elif word in wv.vocab:
                mean.append(wv.syn0norm[wv.vocab[word].index])
                all_words.add(wv.vocab[word].index)
            

        if not mean:
            logging.warning("cannot compute similarity with no input %s", words)
        # FIXME: remove these examples in pre-processing
            return np.zeros(wv.vector_size,)

        mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
        return mean

        def  word_averaging_list(wv, text_list):
            return np.vstack([word_averaging(wv, desc) for desc in text_list ])

        #Tokenize

        import nltk
        def w2v_tokenize_text(text):
            tokens = []
            for sent in nltk.sent_tokenize(text, language='english'):
                for word in nltk.word_tokenize(sent, language='english'):
                    if len(word) < 2:
                        continue
                    tokens.append(word)
            return tokens

        #punkt package download
        nltk.download('punkt')
        test_tokenized = test.apply(lambda r: w2v_tokenize_text(r['EventDescription_parsed']), axis=1).values
        train_tokenized = train.apply(lambda r: w2v_tokenize_text(r['EventDescription_parsed']), axis=1).values

        X_train_word_average = word_averaging_list(wv,train_tokenized)
        X_test_word_average = word_averaging_list(wv,test_tokenized)
    
        #Define and fit the model

        from sklearn import model_selection, naive_bayes, svm 
# fit the training dataset on the classifier
        SVM = svm.SVC(C=15, kernel='linear')
        SVM.fit(X_train_word_average, train['Category'])

# predict the labels on validation dataset
        predictions_SVM = SVM.predict(X_test_word_average)

# Use accuracy_score function to get the accuracy
        print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, test['Category']) * 100)

######################## NEURAL NETWORK
class NeuralNetwork(Resource):
      
    def post(self):
        df_esv = pd.read_csv('esv_EventDescription_tokenized_encoded2.csv') # one hot encoded data , category is first col
        headers = list(df_esv.columns.values)
        # separate into input and output columns
        X = df_esv.iloc[:,1:] # data 1st column is Category(label)
        y = df_esv['Category'] # labels

        train_ratio = 0.7
        validation_ratio = 0.15
        test_ratio = 0.15

        # train is now 75% of the entire data set
        # the _junk suffix means that we drop that variable completely
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_ratio)

        # test is now 10% of the initial data set
        # validation is now 15% of the initial data set
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=test_ratio / (test_ratio + validation_ratio)) 

        # ordinal encode target variable
        label_encoder = LabelEncoder()
        label_encoder.fit(y)
        y_train_enc = label_encoder.transform(y_train)
        y_test_enc = label_encoder.transform(y_test)
        y_val_enc = label_encoder.transform(y_val)
        y_train_enc = to_categorical(y_train_enc)
        y_test_enc = to_categorical(y_test_enc)
        y_val_enc = to_categorical(y_val_enc)

        def create_model(dropout):
           # define the model
           model = Sequential()
           # algo with 76.8% accuracy
           model.add(Dense(256, activation='relu'))
           model.add(Dropout(dropout)) # add dropout
           model.add(Dense(16, activation='softmax'))
           return model

        # parameter values for tuning
        epochs = [8, 15]
        batch_sizes = [128]
        optimizers = ['adam', 'rmsprop']
        dropouts = [0.25, 0.15]
        learning_rates = [0.01, 0.001]
        loss_functions = ['categorical_crossentropy']
        histories = []

        # fine tune
        results = []
        for epoch in epochs:
          for optimizer in optimizers:
            for dropout in dropouts:
              for lr in learning_rates:
                for lf in loss_functions:
                  for bs in batch_sizes:
                    #print("############## Running for following parameters
                    ################")
                    #print("Epoch: ", epoch, ", Optimizer: ", optimizer, ",
                    #Dropout: ", dropout,", Learning rate: ", lr, ", Loss
                    #Function: ", lf)
                    nn = create_model(dropout=dropout)
                    nn.compile(optimizer = optimizer, loss=lf, metrics=['accuracy'])
                    K.set_value(nn.optimizer.learning_rate, lr)
                    history = nn.fit(X_train, y_train_enc, epochs= epoch, batch_size=bs, validation_data = (X_val, y_val_enc), verbose=0)
                    histories.append(history)
                    result = nn.evaluate(X_test, y_test_enc, verbose=0)
                    #print("Run complete.  Accuracy is: ", result[1])
                    results.append({
                        'Epoch': epoch,
                        'BatchSize': bs,
                        'Optimizer': optimizer,
                        'Dropout': dropout,
                        'LearningRate': lr, # todo use learning rate
                        'LossFunction': lf,
                        'Accuracy': result[1],
                        'History' : history
                    })
        results_df = pd.DataFrame(results).sort_values('Accuracy',ascending=False)
        bestaccuracy = results_df.iloc[0]
        besthistory = bestaccuracy['History']
 
########################## DECISION TREE
class DecisionTree(Resource):
      
    def post(self):
        col_names = ['ActionTaken', 'Address', 'AssetLabel', 'CauseCommunity', 'CauseEnvironment', 'CausePre', 'CauseTechnical', 'CauseWorkP', 'ContactType','CorrectProtection','EventDescription','FailedAssets','FailedExplosion','FailedOilFilled','FailedOtherAssets','FailedOtherAssetsOther','FeederNumber','IncidentCause','IncidentConsequence','IncidentDatetime','IncidentFireFFactorReportable','IncidentFireSeverity','IncidentID','IncidentLocationType','IncidentLocationTypeOther','IncidentNumber','IncidentType','Lat','Long','MadeSafe','NetworkType','Status','SubmissionID','SubmittedDateTimeString','Voltage','WeatherStation','Postcode','Locality','Category']
        # load dataset
        pima = pd.read_csv("cleaned_incidents1.csv", header=None, names=col_names)

        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')

        df = df.reset_index(drop=True)
        REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
        BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
        numerical_symbols = re.compile('0-90-9a-z')
 
        STOPWORDS = set(stopwords.words('english'))
 
        def clean_text(text):
             text = text.lower() # lowercase text
             text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text.  substitute the
                                                       # matched string in
                                                                                                             # REPLACE_BY_SPACE_RE
                                                                                                                                                                   # with
                                                                                                                                                                                                                         # space.
             text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text.  substitute the
                                                 # matched string in
                                                                                                 # BAD_SYMBOLS_RE
                                                                                                                                                 # with
                                                                                                                                                                                                 # nothing.
             text = numerical_symbols.sub('', text)
             text = text.replace('x', '')
             #text = re.sub(r'\W+', '', text)
             text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
             return text
        pima['EventDescription'] = pima['EventDescription'].apply(clean_text)
        
        def getLemmText(text):
             tokens = word_tokenize(text)
             lemmatizer = WordNetLemmatizer()
             tokens = [lemmatizer.lemmatize(word) for word in tokens]
             return ' '.join(tokens)
        pima['EventDescription'] = list(map(getLemmText,pima['EventDescription']))

        def getStemmText(text):
             tokens = word_tokenize(text)
             ps = PorterStemmer()
             tokens = [ps.stem(word) for word in tokens]
             return ' '.join(tokens)
        pima['EventDescription'] = list(map(getStemmText,pima['EventDescription']))

        df = pima[['IncidentConsequence','Category','EventDescription']]
        df = df[pd.notnull(df['IncidentConsequence'])]
        df = df[pd.notnull(df['EventDescription'])]
        df.rename(columns = {'IncidentConsequence':'IncidentConsequence_parsed'}, inplace = True)

        #Encoding
        X = pima['EventDescription']
        Y = pima['Category']
        le = LabelEncoder()
        y = le.fit_transform(Y.astype(str))

        pima.dropna(subset= ["Category"], inplace = True)

        # Split dataset into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

        vect = CountVectorizer()
        vect.fit(X_train)
        X_train_dtm = vect.transform(X_train) #Convert a collection of text documents to a matrix of token counts
        X_train_dtm = vect.fit_transform(X_train)
        X_test_dtm = vect.transform(X_test)

        # Create Decision Tree classifer object
        clf = DecisionTreeClassifier()

        # Train Decision Tree Classifer
        clf = clf.fit(X_train_dtm,y_train)

        #Predict the response for test dataset
        y_pred = clf.predict(X_test_dtm)

        acuuracy_score = metrics.accuracy_score(y_test,y_pred) * 100

######################################### RNN
class RNN(Resource):
    def post(self):
        df2 = pd.read_csv('cleaned_incidents1.csv')
        df2.head()

        df = pd.DataFrame()
        df['Category'] = df2.Category
        df['EventDescription'] = df2.EventDescription
        df.dropna()

        df = df.reset_index(drop=True)
        REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
        BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
        STOPWORDS = set(stopwords.words('english'))

        def clean_text(text):
            text = text.lower()
            text = REPLACE_BY_SPACE_RE.sub(' ', text)
            text = BAD_SYMBOLS_RE.sub('', text)
            text = text.replace('x', '')
            text = ' '.join(word for word in text.split() if word not in STOPWORDS)
            return text
        def getLemmText(text):
            tokens = word_tokenize(text)
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(word) for word in tokens]
            return ' '.join(tokens)


        def getStemmText(text):
            tokens = word_tokenize(text)
            ps = PorterStemmer()
            tokens = [ps.stem(word) for word in tokens]
            return ' '.join(tokens)
    
        max_words = 1000 * 5
        max_len = 150 * 1
        tok = Tokenizer(num_words=max_words,filters = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True, split=" ",)
        tok.fit_on_texts(X_train)
        sequences = tok.texts_to_sequences(X_train)
        sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)
        units_mul = 100 * 2
        unique_categories = np.unique(Y)
        labels_count = len(unique_categories)
        dropout = 0.2
    
        def RNN3():
            model = tf.keras.Sequential()
            model.add(Embedding(max_words, units_mul, input_length=sequences_matrix.shape[1]))
            model.add(tf.keras.layers.SpatialDropout1D(dropout))
            model.add(LSTM(units_mul, dropout=dropout, recurrent_dropout=dropout))
            model.add(Dense(units_mul, activation='relu'))
            model.add(Dropout(dropout))
            model.add(Dense(labels_count, activation='softmax'))
            return model

        df['EventDescription'] = df['EventDescription'].apply(clean_text)
        df['EventDescription'] = list(map(getLemmText,df['EventDescription']))
        df['EventDescription'] = list(map(getStemmText,df['EventDescription']))

        X = df.EventDescription
        Y = df.Category
        le = LabelEncoder()
        Y = le.fit_transform(Y.astype(str))

        X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.15, stratify=Y)

        model = RNN3()

        optimiser = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimiser,
                  metrics=['accuracy'])

        with tf.device('/device:GPU:0'):
            history = model.fit(sequences_matrix,Y_train, batch_size=128,
                        epochs=10, validation_split=0.2
                        ,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0, patience=7)])

        test_sequences = tok.texts_to_sequences(X_test)
        test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
        accr = model.evaluate(test_sequences_matrix,Y_test)
        print(accr[1])
  
######################################### LR
class LogisticRegression(Resource):

    def post(self):
        df = pd.read_csv(r'C:\Users\Admin\Documents\d2i---deakin-energy_2020t2\Datasets\ESV Data\cleaned_incidents1.csv')
        df = df[['EventDescription','Category']]
        df = df[pd.notnull(df['EventDescription'])]
        df.rename(columns = {'EventDescription':'EventDescription_parsed'}, inplace = True)

        df.dropna(subset= ["Category"], inplace = True)

        df.index = range(6489)
        df['EventDescription_parsed'].apply(lambda x: len(x.split(' ')))

        def print_incident(index):
            example = df[df.index == index][['EventDescription_parsed', 'Category']].values[0]    
            if len(example) > 0:
                print(example[0])       
                print('Category:', example[1])


        #Remove punctuations
        punctuation_signs = list("?:!.,;")

        for punct_sign in punctuation_signs:
            df['EventDescription_parsed'].str.replace(punct_sign, '')
        df.loc[1]['EventDescription_parsed']

        def cleanText(text):

            punctuation_signs = list("?:!.,;")
            text = text.lower()
            for punct_sign in punctuation_signs:
                text = text.replace(punct_sign, '')
            return text

        df['EventDescription_parsed'] = df['EventDescription_parsed'].apply(cleanText)
        
        #Train|Test split
        train, test = train_test_split(df, test_size=0.3, random_state = 42)


        #Feeding GoogleNewsVector

        
        wv = gensim.models.KeyedVectors.load_word2vec_format(r"C:\Users\Admin\Documents\d2i---deakin-energy_2020t2\Data Analysis\Deliverable 3\GoogleNews-vectors-negative300.bin.gz", binary=True)
        wv.init_sims(replace=True)

        def word_averaging(wv, words):
            all_words, mean = set(), []
    
    
            for word in words:
                if isinstance(word, np.ndarray):
                    mean.append(word)
            
                elif word in wv.vocab:
                    mean.append(wv.syn0norm[wv.vocab[word].index])
                    all_words.add(wv.vocab[word].index)
            

            if not mean:
                logging.warning("cannot compute similarity with no input %s", words)
            # FIXME: remove these examples in pre-processing
                return np.zeros(wv.vector_size,)

            mean = gensim.matutils.unitvec(np.array(mean).mean(axis=0)).astype(np.float32)
            return mean

        def  word_averaging_list(wv, text_list):
            return np.vstack([word_averaging(wv, desc) for desc in text_list ])

        #Tokenize

        
        def w2v_tokenize_text(text):
            tokens = []
            for sent in nltk.sent_tokenize(text, language='english'):
                for word in nltk.word_tokenize(sent, language='english'):
                    if len(word) < 2:
                        continue
                    tokens.append(word)
            return tokens

        #punkt package download
        
        test_tokenized = test.apply(lambda r: w2v_tokenize_text(r['EventDescription_parsed']), axis=1).values
        train_tokenized = train.apply(lambda r: w2v_tokenize_text(r['EventDescription_parsed']), axis=1).values

        X_train_word_average = word_averaging_list(wv,train_tokenized)
        X_test_word_average = word_averaging_list(wv,test_tokenized)

        logreg = LogisticRegression(n_jobs=1, C=20, max_iter=1000)
        logreg = logreg.fit(X_train_word_average, train['Category'])
        y_pred = logreg.predict(X_test_word_average)
        print('accuracy %s' % accuracy_score(y_pred, test.Category))

api.add_resource(NLP,'/') 
api.add_resource(KNN, '/')
api.add_resource(SVM,'/')
api.add_resource(NeuralNetwork,'/')
api.add_resource(DecisionTree,'/')
api.add_resource(RNN,'/')
api.add_resource(LogisticRegression,'/')

if __name__ == '__main__':
    
    HOST = os.environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(os.environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    app.run(HOST, PORT, threaded=True)
