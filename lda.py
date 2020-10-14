#!/usr/bin/env python
# coding: utf-8

# In[11]:


import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# 0. Load in the data and split the descriptive and the target feature
df = pd.read_csv('diabetes.csv')

#tuple data
df.column=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness','Insulin', 'BMI','DiabetesPedigreeFunction','Age', 'Outcome']
df

X = df.iloc[:,0:4].values
y = df.iloc [:,4].values

target = df['Pregnancies'].values

#Instantiate the method and fit_transform the algotithm
pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

lda =LinearDiscriminantAnalysis(n_components=2) # The n_components key word gives us the projection to the n most discriminative directions in the dataset. We set this parameter to two to get a transformation in two dimensional space
X_r2 = lda.fit (X,y).transform(X)

print ('explained variance ratio (first two components): %s'
        %str(pca.explained_variance_ratio_))

#PLot the data
plt.figure()
colors = ['navy', 'turquoise', 'darkorange']
lw = 2

for color, i, target in zip(colors, [0,1,2], target):
    plt.scatter(X_r[y==i,0], X_r[y==i,1], color=color, alpha=.8, lw=lw, label=target)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of IRIS dataset')
    
plt.figure()
for color, i, target in zip(colors, [0,1,2], target):
    plt.scatter(X_r2[y==i, 0], X_r2[y==i,1], alpha=.8, color=color, label=target)
plt.legend(loc='best', shadow=False, scatterpoints =1)
plt.title('LDA of apa hayo :')

plt.show()


# In[4]:





# In[ ]:




