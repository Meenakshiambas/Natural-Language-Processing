#!/usr/bin/env python
# coding: utf-8

# # Natural Language Processing

# ## Importing the libraries

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ## Importing the dataset

# In[2]:


dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)


# In[3]:


dataset.head(100)


# ## Cleaning the texts

# In[4]:


import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-z]', ' ' , dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stop_words = stopwords.words('english')
    all_stop_words.remove('not')
    review =[ps.stem(word) for word in review if not word in set(all_stop_words)]
    review = ' '.join(review)
    corpus.append(review)


# In[5]:


print(corpus)


# ## Creating the Bag of Words model

# In[6]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values


# In[7]:


len(X[0])


# ## Splitting the dataset into the Training set and Test set

# In[8]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# ## Training the Naive Bayes model on the Training set

# In[9]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)


# ## Predicting the Test set results

# In[10]:


y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# ## Making the Confusion Matrix

# In[11]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)


# # Predicting if a single review is positive or negative
# 

# # Positive review
# Use our model to predict if the following review:
# 
# "I love this restaurant so much"
# 
# is positive or negative.
# 
# solution: We just repeat the same text preprocessing process we did before, but this time with a single review.

# In[12]:


new_review = 'I love this restaurant so much'
new_review = re.sub('[^a-zA-Z]', ' ', new_review)
new_review = new_review.lower()
new_review = new_review.split()
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
new_review = ' '.join(new_review)
new_corpus = [new_review]
new_X_test = cv.transform(new_corpus).toarray()
new_y_pred = classifier.predict(new_X_test)
print(new_y_pred)


# The review was correctly predicted as positive by our model.

# # Negative review

# Use our model to predict if the following review:
# 
# "I hate this restaurant so much"
# 
# is positive or negative.
# 
# Solution: We just repeat the same text preprocessing process we did before, but this time with a single review.
# 
# 

# In[13]:


new_review = 'I hate this restaurant so much'
new_review = re.sub('[^a-zA-Z]', ' ', new_review)
new_review = new_review.lower()
new_review = new_review.split()
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
new_review = ' '.join(new_review)
new_corpus = [new_review]
new_X_test = cv.transform(new_corpus).toarray()
new_y_pred = classifier.predict(new_X_test)
print(new_y_pred)


# The review was correctly predicted as negative by our model.

# In[ ]:





# In[ ]:




