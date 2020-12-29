#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np # linear algebra
import pandas as pd 
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.lancaster import LancasterStemmer
#from nltk.stem.porter import PorterStemmer
stop_words = set(stopwords.words('english'))


# In[32]:


train=pd.read_csv("C:/Users/HP/Downloads/train_2kmZucJ.csv")
test=pd.read_csv("C:/Users/HP/Downloads/test_oJQbWVk.csv")
print(len(train.index))
print(len(test.index))
train.head()


# In[33]:


combi=train.append(test, ignore_index=True)
combi['text']=combi['tweet'].str.lower()
combi.head()


# In[34]:


import re
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt 


# In[35]:


#combi['tidy_tweet'] = np.vectorize(remove_pattern(combi['text'], "(\w+:\/\/\S+)"))
#combi.head()


# In[36]:


combi['tidy_tweet'] = combi['text'].str.replace("[^a-zA-Z]", " ")
combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if not w in stop_words]))
combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split())

stemmer = LancasterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
print(tokenized_tweet.head())


# In[37]:


print(len(tokenized_tweet))


# In[38]:


for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])

combi['tidy_tweet'] = tokenized_tweet
combi.head()


# In[40]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
# TF-IDF feature matrix
tfidf = tfidf_vectorizer.fit_transform(combi['tidy_tweet'])
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

train_tfidf = tfidf[:7920,:]
test_tfidf = tfidf[7920:,:]


# splitting data into training and validation set
xtrain_tfidf, xvalid_tfidf, ytrain, yvalid = train_test_split(train_tfidf, train['label'], random_state=42, test_size=0.3)

xtrain_tfidf = train_tfidf[ytrain.index]
xvalid_tfidf = train_tfidf[yvalid.index]

lreg=LogisticRegression()
lreg.fit(xtrain_tfidf, ytrain)

prediction = lreg.predict_proba(xvalid_tfidf)
prediction_int = prediction[:,1] >= 0.3
prediction_int = prediction_int.astype(np.int)


# In[42]:


test_pred = lreg.predict_proba(test_tfidf)
test_pred_int = test_pred[:,1] >= 0.3
test_pred_int = test_pred_int.astype(np.int)
test['label'] = test_pred_int
submission = test[['id','label']]
submission.to_csv('sub_lreg_bow%.csv', index=False) # writing data to a CSV file


# In[ ]:




