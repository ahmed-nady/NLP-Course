#"https://www.kaggle.com/balakishan77/spam-or-ham-email-classification"

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
	
corpus_path = "spam-filter/emails.csv"
corpus  = None
preprocessed_corpus = None

MultinomialNB_classifier = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
cv = CountVectorizer()
tfidf_vectorizer = TfidfVectorizer()

def read_corpus(corpus_path):

	global corpus
	corpus = pd.read_csv(corpus_path)
	#Checking class distribution
	print("Checking class distribution", corpus.groupby('spam').count())


def preprocess_text(corpus):
	global preprocessed_corpus
	#Using Natural Language Processing to cleaning the text to make one corpus
	# Cleaning the texts
	import re
	import nltk
	from nltk.corpus import stopwords
	from nltk.stem.porter import PorterStemmer
	corpus['tokens'] = corpus['text'].map(lambda text:  nltk.tokenize.word_tokenize(text)) 
	stop_words = set(nltk.corpus.stopwords.words('english'))
	corpus['filtered_text'] = corpus['tokens'].map(lambda tokens: [w for w in tokens if not w in stop_words]) 
	#Every mail starts with 'Subject :' lets remove this from each mail 

	corpus['filtered_text'] = corpus['filtered_text'].map(lambda text: text[2:])
	#Mails still have many special charater tokens which may not be relevant for spam filter, lets remove these
	#Joining all tokens together in a string
	corpus['filtered_text'] = corpus['filtered_text'].map(lambda text: ' '.join(text))
	#removing apecial characters from each mail 
	corpus['filtered_text'] = corpus['filtered_text'].map(lambda text: re.sub('[^A-Za-z0-9]+', ' ', text))

	wnl = nltk.WordNetLemmatizer()
	corpus['filtered_text'] = corpus['filtered_text'].map(lambda text: wnl.lemmatize(text))

	print("corpus['filtered_text'][4] \n",corpus['filtered_text'][4]) 
	print("\ncorpus['text'][4] \n",corpus['text'][4]) 
	 
def extract_features():
	global corpus,cv,tfidf_vectorizer
	# Creating the Bag of Words model
	X = cv.fit_transform(corpus['filtered_text'].values)
	#X = tfidf_vectorizer.fit_transform(corpus['filtered_text'].values)
	y = corpus['spam'].values
	print(X.shape,y.shape)
	return X,y

def split_corpus(X,y):
	#print("X.shape,y.shape",X.shape,y.shape)
	# Splitting the dataset into the Training set and Test set
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
	return X_train, X_test, y_train, y_test
def train_MultinomialNB_model(X_train,y_train):
	global MultinomialNB_classifier
	# Fitting Naive Bayes classifier to the Training set
	MultinomialNB_classifier.fit(X_train , y_train)

def predict():
	global MultinomialNB_classifier,cv,tfidf_vectorizer
	#Predictions on sample text
	examples = ['cheap Viagra', "Forwarding you minutes of meeting"]

	example_counts = cv.transform(examples)
	#example_tfidf = tfidf_vectorizer.transform(examples)
	predictions = MultinomialNB_classifier.predict(example_counts)
	print("Prediction on test email: ",predictions)
	# Predicting the Test set results

def calculate_confusion_matrix(y_test):
	global MultinomialNB_classifier
	# Making the Confusion Matrix
	from sklearn.metrics import confusion_matrix
	y_pred = MultinomialNB_classifier.predict(X_test)
	cm = confusion_matrix(y_test, y_pred)
	print("confusion matrix: ",cm)
	'''
	Confusion Matrix
	array([[863,  11],
	       [  1, 264]])
	'''
def calculate_model_accuracy(y_test):
	global MultinomialNB_classifier
	#this function computes subset accuracy
	from sklearn.metrics import accuracy_score
	from sklearn.metrics import precision_score
	y_pred = MultinomialNB_classifier.predict(X_test)
	model_acc = accuracy_score(y_test, y_pred) #0.9894644424934153
	model_acc2 = accuracy_score(y_test, y_pred,normalize=False) #1129 out of 1139
	model_prec = precision_score(y_test, y_pred)
	print("Model Accuracy: ",model_acc,model_acc,model_prec)
 

if __name__ =='__main__':
	ans= True

	X,y,X_train, X_test, y_train, y_test = None,None,None,None,None,None
	y_pred = None
	while ans:
		print(""" \n
		1.Reading emails corpus
		2.preprocess text using nltk
		3.Feature extraction
		4-train_MultinomialNB_model (Navive bayes classifier)
		5. predict
		6.calculate_confusion_matrix
		7- calculate_model_accuracy
		8.Exit/Quit
	    """)
		ans=input("What would you like to do? ")
		
		if ans=="1":
			read_corpus(corpus_path)
			print("Reading emails corpus: ",corpus)
		elif ans=="2":
			preprocess_text(corpus)
			print("preprocess text using nltk",preprocessed_corpus)
		elif ans=="3":
			X,y = extract_features()
			print("Features are extracted")
		elif ans=="4":
			
			X_train, X_test, y_train, y_test = split_corpus(X,y)
			print(X_train.shape,y_train.shape)
			print("train_MultinomialNB_model",train_MultinomialNB_model(X_train,y_train))
		elif ans=="5":
			predict()
			#print("Number of mislabeled emails out of a total %d emails : %d" % (X_test.shape[0], (y_test != y_pred).sum()))
			#print("prediction: ",y_pred)
		elif ans=="6":
			print("calculate_confusion_matrix",calculate_confusion_matrix(y_test))
		elif ans=="7":
			print("calculate_model_accuracy",calculate_model_accuracy(y_test))
		elif ans=="8":
			ans=False
		else:
			print("Please select one of availale option")
