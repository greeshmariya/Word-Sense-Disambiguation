import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, accuracy_score

traindoc1=open("apple-fruit.txt",encoding='utf8').read()
traindoc2=open("apple-computers.txt",encoding="utf8").read()
testdoc1= open("test_fruit.txt",'r').read()
testdoc2= open("test_computer.txt",encoding='utf8').read()

dataframe   = [[traindoc1, 'fruit'],
               [traindoc2, 'computer'],
               [testdoc1,'fruit'],
               [testdoc2,'computer']]
                 
corpus= pd.DataFrame(columns=['text','label'],data=my_python_list)

Encoder = LabelEncoder()
encode = Encoder.fit_transform(corpus['label'])

corpus['text'].dropna(inplace=True)
corpus['text'] = [entry.lower() for entry in corpus['text']]
corpus['text']= [word_tokenize(entry) for entry in corpus['text']]

tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

for index,entry in enumerate(corpus['text']):
    words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    corpus.loc[index,'text_final'] = str(words)

#print(Corpus['text_final'].head())

Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(corpus['text_final'],corpus['label'],test_size=0.5)


Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(corpus['text_final'])

Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,Train_Y)

predictions_NB = Naive.predict(Test_X_Tfidf)

print("Naive Bayes Accuracy Score -> ",accuracy_score(predictions_NB, Test_Y)*100)

SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)

predictions_SVM = SVM.predict(Test_X_Tfidf)

print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)
print(confusion_matrix(predictions_SVM, Test_Y))
