import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import gensim ,re
from gensim.models import Word2Vec
import scipy

import nltk
#nltk.download('all')
from nltk.tokenize import sent_tokenize, word_tokenize 
import warnings 
import sklearn




patent = pd.ExcelFile("D:\python patents\Patent_similarity.xlsx")
patent.sheet_names
df1= patent.parse("Sheet2")
df2= patent.parse("Sheet1")

df1.columns

dftrain= df1[['Independent Claims','CPC']].astype(str)
dftest =df2[['Independent Claims','CPC']].astype(str)

#dftrain['Independent Claims']=dftrain['Independent Claims'].apply(default_clean)
from nltk.corpus import stopwords
stop = stopwords.words('english')
dftrain['Independent Claims'] = dftrain['Independent Claims'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
dftrain['Independent Claims'].head()

dftrain['Independent Claims'] = dftrain['Independent Claims'].str.replace('[^\w\s]','')
dftrain['Independent Claims'].head()

dftrain['Independent Claims'] = dftrain['Independent Claims'].apply(lambda x: " ".join(x.lower() for x in x.split()))
dftrain['Independent Claims'].head()

dftrain['Independent Claims'] = dftrain['Independent Claims'].apply(lambda x: " ".join(x for x in x.split() if not x.isdigit()))
dftrain['Independent Claims'].head()

from textblob import Word    ###lemmatiztaion similar to stemming
dftrain['Independent Claims'] = dftrain['Independent Claims'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
dftrain['Independent Claims'].head()

#### for the test dat set data cleaning

stop = stopwords.words('english')
dftest['Independent Claims'] = dftest['Independent Claims'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
dftest['Independent Claims'].head()

dftest['Independent Claims'] = dftest['Independent Claims'].str.replace('[^\w\s]','')
dftest['Independent Claims'].head()

dftest['Independent Claims'] = dftest['Independent Claims'].apply(lambda x: " ".join(x.lower() for x in x.split()))
dftest['Independent Claims'].head()

dftest['Independent Claims'] = dftest['Independent Claims'].apply(lambda x: " ".join(x for x in x.split() if not x.isdigit()))
dftest['Independent Claims'].head()

dftest['Independent Claims'] = dftest['Independent Claims'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
dftest['Independent Claims'].head()


from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
#from nltk import word_tokenize


class TaggedDocumentIterator():
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield TaggedDocument(words=doc.split(), tags=[self.labels_list[idx]])
 
docLabels = list(dftrain['Independent Claims'])
data = list(dftrain['Independent Claims'])
sentences = TaggedDocumentIterator(data, docLabels)

doclabels_test=list(dftest['CPC'])
data_test = list(dftest['Independent Claims'])
sentences_test = TaggedDocumentIterator(data, docLabels)



model = Doc2Vec(size=100, window=10, min_count=5, workers=11,alpha=0.025, iter=20)
model.build_vocab(sentences)
model.train(sentences_test,total_examples=model.corpus_count, epochs=model.iter)



# Store the model to mmap-able files
model.save('model_docsimilarity.doc2vec')
# Load the model
model = Doc2Vec.load('model_docsimilarity.doc2vec')


   #Convert the sample document into a list and use the infer_vector method to get a vector representation for it
    #new_doc_words = dftest['Independent Claims']
    #new_doc_vec = model.infer_vector(new_doc_words, steps=50, alpha=0.1)
 
    #use the most_similar utility to find the most similar documents.
   # similars = model.docvecs.most_similar(positive=[new_doc_vec],topn=10)
    #np.savetxt("similarityscore.csv", similars, delimiter=",", fmt='%s')

    
#print(similars) #### This overal patent similarit of the all documents

from itertools import islice
import sys

for col in  dftest['Independent Claims']:
   # print(col)
    new_doc_words1= col
    new_doc_vec1 = model.infer_vector(new_doc_words1, steps=50, alpha=0.1)
    similars1 = model.docvecs.most_similar(positive=[new_doc_vec1],topn=10)
    print(col,similars1,file=open("givenformat.csv","a",encoding="utf-8"))  
    
    
final_print = pd.read_csv("D:\python patents\givenformat.csv")

final_print.iloc[:,0]=dftest.iloc[:,0]
final_print.head()
final_print
