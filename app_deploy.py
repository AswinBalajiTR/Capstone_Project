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


def load_model():
    with open('finalized_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data


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


def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [x for x in words if x not in stop_words]


def convert_lower(text):
    return text.lower()


def lemmatize_word(text):

    wordnet = WordNetLemmatizer()
    return " ".join([wordnet.lemmatize(word) for word in text])


stop = set(stopwords.words('english'))


def wordcloud_draw(dataset, color='white'):
    words = ' '.join(dataset)
    cleaned_word = ' '.join(
        [word for word in words.split()if (word != 'news' and word != 'text')])
    wordcloud = WordCloud(stopwords=stop, background_color=color,
                          width=6000, height=2000).generate(cleaned_word)
    plt.figure(1, figsize=(10, 7))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()


data = load_model()

save = st.button('Bar Chart')
word = st.button('Word Cloud')

test2 = pd.DataFrame()
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
exec = pd.DataFrame(data.predict(x))
test2['Category'] = pd.DataFrame(exec)
test2['Category'] = test2['Category'].replace(0, "Sport")
test2['Category'] = test2['Category'].replace(1, "Tech")
test2['Category'] = test2['Category'].replace(2, "Politics")
test2['Category'] = test2['Category'].replace(3, "Entertainment")
test2['Category'] = test2['Category'].replace(4, "Business")
business = test2[test2['Category'] == "Business"]
tech = test2[test2['Category'] == "Tech"]
politics = test2[test2['Category'] == "Politics"]
sport = test2[test2['Category'] == "Sport"]
entertainment = test2[test2['Category'] == "Entertainment"]

if(save):
    # fig = plt.figure(figsize=(15, 5))
    # test2['Category'].value_counts().plot(
    #     kind="bar", color=["darkblue", "lightgreen", "skyblue", "green", "turquoise"])
    # plt.xlabel("News Category")
    # plt.ylabel("News Count")
    # plt.title("Total number of news in each category")
    # plt.show()
    # fig1, ax1 = plt.subplots()
    # colors = ["darkblue", "lightgreen", "skyblue", "green", "turquoise"]
    # count = [business['Category'].count(), tech['Category'].count(), politics['Category'].count(),
    #          sport['Category'].count(), entertainment['Category'].count()]
    # ax1.pie(count, labels=['Business', 'Tech', 'Politics', 'Sport', 'Entertainment'],
    #         autopct="%1.1f%%",
    #         shadow=True,
    #         colors=colors,
    #         startangle=45,
    #         explode=(0.05, 0.05, 0.05, 0.05, 0.05))
    # ax1.axis('equal')
    # st.pyplot(fig1)
    st.bar_chart(test2['Category'].value_counts())

if(word):
    # if(save):
    #     id_sheet = "1jE3rlC4D_1yfeuhS9ihatXpdnI4m2y3BLbkrLQvc8j4"
    #     excel = pd.ExcelFile(
    #         f"https://docs.google.com/spreadsheets/d/{id_sheet}/export?format=xlsx")
    #     test = pd.read_excel(excel, "Sheet1", header=0)
    #     test2 = test
    #     test['Text'] = test['Text'].apply(remove_tags)
    #     test['Text'] = test['Text'].apply(special_char)
    #     test['Text'] = test['Text'].apply(convert_lower)
    #     test['Text'] = test['Text'].apply(remove_stopwords)
    #     test['Text'] = test['Text'].apply(lemmatize_word)
    #     X = pd.DataFrame(test['Text'])
    #     x = np.array(X.iloc[:, 0].values)
    #     cv = CountVectorizer(max_features=5000)
    #     x = cv.fit_transform(X.Text).toarray()
    #     exec = pd.DataFrame(data.predict(x))
    #     test2['Category'] = pd.DataFrame(exec)
    #     test2['Category'] = test2['Category'].replace(0, "sport")
    #     test2['Category'] = test2['Category'].replace(1, "tech")
    #     test2['Category'] = test2['Category'].replace(2, "politics")
    #     test2['Category'] = test2['Category'].replace(3, "entertainment")
    #     test2['Category'] = test2['Category'].replace(4, "business")
    business = business['Text']
    tech = tech['Text']
    politics = politics['Text']
    sport = sport['Text']
    entertainment = entertainment['Text']

    st.write("Business related words:")
    wordcloud_draw(business, 'white')
    st.write("Tech related words:")
    wordcloud_draw(tech, 'white')
    st.write("Politics related words:")
    wordcloud_draw(politics, 'white')
    st.write("Sport related words:")
    wordcloud_draw(sport, 'white')
    st.write("Entertainment related words:")
    wordcloud_draw(entertainment, 'white')

st.subheader("SELECT NEWS CATEGORY : ")

bn = st.checkbox('BUSINESS')
tn = st.checkbox('TECH')
pn = st.checkbox('POLITICAL')
sn = st.checkbox('SPORTS')
en = st.checkbox('ENTERTAINMENT')

okay = st.button('Read News')
if(okay):
    test3 = pd.DataFrame()
    a = []
    if(bn):
        a.append("Business")
    if(tn):
        a.append("Tech")
    if(pn):
        a.append("Politics")
    if(sn):
        a.append("Sport")
    if(en):
        a.append("Entertainment")
    for i in a:
        test3 = pd.concat([test3, test2[test2['Category'] == i]])
    st.write(test3)

# test1 = np.array(test.iloc[:, 0].values)
# cv = CountVectorizer(max_features=5000)
# test1 = cv.fit_transform(test.Text).toarray()
# exec = pd.DataFrame(data.predict(test1))
# out = pd.concat([test, exec])
# print(out)
