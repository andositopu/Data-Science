#!/usr/bin/env python
# coding: utf-8

# ## Classification with NBC

# ### Import libraries

# In[1]:


import pandas as pd


# In[2]:


# read file into pandas using a relative path
path = 'data_penyakit.xlsx'
dataset = pd.read_excel(path)


# In[3]:


# examine the shape
dataset.shape


# In[4]:


# examine the first 5 rows
dataset.head()


# In[5]:


# examine the class distribution
dataset.label.value_counts()


# In[6]:


# convert label to a numerical variable
dataset['label_num'] = dataset.label.map({'Muntaber':0, 'Demam Berdarah':1, 'Influenza':2})


# In[7]:


# check that the conversion worked
dataset.head(10)


# In[8]:


# how to define X and y (from the SMS data) for use with COUNTVECTORIZER
X = dataset[['Muntah', 'Batuk', 'Beringus', 'Pusing']]
y = dataset['label_num']
print(X.shape)
print(y.shape)


# In[9]:


# split X and y into training and testing sets
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:





# Untuk merubah kata kedalam angka bisa dengan 2 cara

# In[10]:


# import and instantiate a Multinomial Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()


# In[11]:


# train the model using X_train_dtm (timing it with an IPython "magic command")
get_ipython().run_line_magic('time', 'nb.fit(X_train, y_train)')


# In[12]:


# make class predictions for X_test_dtm
y_pred_class = nb.predict(X_test)


# In[13]:


# calculate accuracy of class predictions
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)


# In[14]:


# print the confusion matrix
metrics.confusion_matrix(y_test, y_pred_class)


# In[15]:


# calculate AUC
# metrics.roc_auc_score(y_test, y_pred_prob)


# In[16]:


#Test Data Baru
# 0 = Demam Berdarah
# 1 = Influenza
# 2 = Muntaber
new_data =[[1,0,1,1],
          [0,1,1,1]]

hasil=nb.predict(new_data)

for i in hasil:
    if i == 0:
        print("Pasien diprediksi terjangkit 'Demam Berdarah'")
    elif i == 1:
        print("Pasien diprediksi terjangkit 'Influenza'")
    else:
        print("Pasien diprediksi terjangkit 'Muntaber'")


# In[ ]:




