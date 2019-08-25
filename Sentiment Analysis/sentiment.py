import pandas as pd
import re
import warnings
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import  classification_report
warnings.filterwarnings('ignore')


dataset = pd.read_csv('Restaurent.csv')
corpus = []

def data_preprocessing():
    for i in range(dataset.shape[0]):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        corpus.append(review)

def naiveBayes():
    classifier = MultinomialNB()
    classifier.fit(X_train, Y_train)
    pred = classifier.predict(X_test)
    return pred

def decisionTree():
    tree = DecisionTreeClassifier()
    tree.fit(X_train, Y_train)
    pred = tree.predict(X_test)
    return pred

def svm():
    svc = SVC(kernel='linear')
    svc.fit(X_train, Y_train)
    pred = svc.predict(X_test)
    return pred

def logisticRegresion():
    logistic = LogisticRegression()
    logistic.fit(X_train, Y_train)
    pred = logistic.predict(X_test)
    return pred

data_preprocessing()
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
Y = dataset['Liked']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=True)

print('Classification Report for Naive Bayes')
print(classification_report(Y_test, naiveBayes()))
print('Classification Report for Decision Tree')
print(classification_report(Y_test, decisionTree()))
print('Classification Report for SVM')
print(classification_report(Y_test, svm()))
print('Classification Report for Logistic Regression')
print(classification_report(Y_test, logisticRegresion()))