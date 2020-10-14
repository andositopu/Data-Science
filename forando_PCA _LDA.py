#!/usr/bin/env python
# coding: utf-8

# # PCA

# In[1]:


import numpy as np # untuk matriks
#untuk manipulasi data
import pandas as pd

from matplotlib import pyplot as plt # untuk plot grafik

from mpl_toolkits.mplot3d import Axes3D

# Load data set and display first few observations
dataset = pd.read_csv("diabetes.csv")
dataset.head()


# In[3]:


# Summary statistics (mean, stdev, min, max)
dataset.describe()


# In[4]:


# Define features
X = dataset.iloc[:,0:8]

# Define categorical outcome 
y = dataset.iloc[:,8]

# Standardize feature space to have mean 0 and variance 1
X_std = (X-np.mean(X,axis = 0))/np.std(X,axis = 0)


# In[5]:


# Step 1: Find covariance matrix of X

# Obtain covariance matrix for X (note columns are the features)
cov_matrix = np.cov(X_std, rowvar=False)
# Note that covariance matrix is 8x8 since their are 8 features
print('Covariance matrix of X: \n%s' %cov_matrix)


# In[6]:


# Step 2: Obtain eigenvectors and eigenvalues

# Obtain eigenvalues and eigenvectors 
eigenvalues, eigenvectors, = np.linalg.eig(cov_matrix)

# eigenvectors is a 8x8 matrix
print('Eigenvectors of Cov(X): \n%s' %eigenvectors)

# eigenvalues is a 8x1 vector
print('\nEigenvalues of Cov(X): \n%s' %eigenvalues)


# In[7]:


# Step 3 (continued): Sort eigenvalues in descending order

# Make a set of (eigenvalue, eigenvector) pairs
eig_pairs = [(eigenvalues[index], eigenvectors[:,index]) for index in range(len(eigenvalues))]

# Sort the (eigenvalue, eigenvector) pairs from highest to lowest with respect to eigenvalue
eig_pairs.sort()
eig_pairs.reverse()

# Extract the descending ordered eigenvalues and eigenvectors
eigvalues_sort = [eig_pairs[index][0] for index in range(len(eigenvalues))]
eigvectors_sort = [eig_pairs[index][1] for index in range(len(eigenvalues))]

# Let's confirm our sorting worked, print out eigenvalues
print('Eigenvalues in descending order: \n%s' %eigvalues_sort)


# In[8]:


# Find cumulative variance of each principle component
var_comp_sum = np.cumsum(eigvalues_sort)/sum(eigvalues_sort)

# Show cumulative proportion of varaince with respect to components
print("Cumulative proportion of variance explained vector: \n%s" %var_comp_sum)

# x-axis for number of principal components kept
num_comp = range(1,len(eigvalues_sort)+1)

# Chart title
plt.title('Cum. Prop. Variance Explain and Components Kept')

# x-label
plt.xlabel('Principal Components')

# y-label
plt.ylabel('Cum. Prop. Variance Expalined')

# Scatter plot of cumulative variance explained and principal components kept
plt.scatter(num_comp, var_comp_sum)

# Show scattor plot
plt.show()


# In[9]:


# Step 4: Project data onto 2d 

# Keep the first two principal components 
# P_reduce is 8 x 2 matrix
P_reduce = np.array(eigvectors_sort[0:2]).transpose()

# Let's project data onto 2D space
# The projected data in 2D will be n x 2 matrix
Proj_data_2D = np.dot(X_std,P_reduce)


# In[10]:


# Visualize data in 2D

# Plot projected the data onto 2D (test negative for diabetes)
negative = plt.scatter(Proj_data_2D[:,0][y == 0], Proj_data_2D[:,1][y == 0])

# Plot projected the data onto 2D (test positive for diabetes)
positive = plt.scatter(Proj_data_2D[:,0][y == 1], Proj_data_2D[:,1][y == 1], color = "red")


# Chart title
plt.title('PCA Dimensionality Reduction to 2D')

# y-label
plt.ylabel('Principal Component 2')

# x-label
plt.xlabel('Principal Component 1')

# legend
plt.legend([negative,positive],["No Diabetes", "Have Diabetes"])

# Show scatter plot
plt.show()


# # LDA

# In[2]:


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


# In[ ]:




