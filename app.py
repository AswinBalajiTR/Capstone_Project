from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import make_scorer, roc_curve, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from json import load
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


st.title("NEWS CLASSIFICATION AND RECOMMENDATION")


def remove_tags(text):
    remove = re.compile(r'')
    return re.sub(remove, '', text)


def special_char(text):
    reviews = ''
    for x in text:
        if x.isalnum():
            reviews = reviews + x
        else:
            reviews = reviews + ' '
    return reviews


def convert_lower(text):
    return text.lower()


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [x for x in words if x not in stop_words]


def lemmatize_word(text):
    wordnet = WordNetLemmatizer()
    return " ".join([wordnet.lemmatize(word) for word in text])


def load_model():
    with open('finalized_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data


data = load_model()

save = st.button('Update')
test2 = pd.DataFrame()

if(save):
    id_sheet = "1jE3rlC4D_1yfeuhS9ihatXpdnI4m2y3BLbkrLQvc8j4"
    excel = pd.ExcelFile(
        f"https://docs.google.com/spreadsheets/d/{id_sheet}/export?format=xlsx")
    test = pd.read_excel(excel, "Sheet1", header=0)
    test2 = test
    test['Text'] = test['Text'].apply(remove_tags)
    test['Text'] = test['Text'].apply(special_char)
    test['Text'] = test['Text'].apply(convert_lower)
    test['Text'] = test['Text'].apply(remove_stopwords)
    test['Text'] = test['Text'].apply(lemmatize_word)
    X = pd.DataFrame(test['Text'])
    x = np.array(X.iloc[:, 0].values)
    cv = CountVectorizer(max_features=5000)
    x = cv.fit_transform(X.Text).toarray()
    exec = data.predict(x)
    exec = list(exec)
    for i in range(len(exec)):
        if exec[i] == 0:
            exec[i] = "Business News"
        elif exec[i] == 1:
            exec[i] = "Tech News"
        elif exec[i] == 2:
            exec[i] = "Politics News"
        elif exec[i] == 3:
            exec[i] = "Sports News"
        elif exec[i] == 4:
            exec[i] = "Entertainment News"
    test2['Category'] = pd.DataFrame(exec)
    st.write(test2)

    st.subheader('Select news category : ')

# bn = st.checkbox('Business')
# tn = st.checkbox('Tech')
# pn = st.checkbox('Political')
# sn = st.checkbox('Sports')
# en = st.checkbox('Entertainment')
    okay = st.button('Get News')

    if(okay):
        st.write(test2)
# test1 = np.array(test.iloc[:, 0].values)
# cv = CountVectorizer(max_features=5000)
# test1 = cv.fit_transform(test.Text).toarray()
# exec = pd.DataFrame(data.predict(test1))
# out = pd.concat([test, exec])
# print(out)
