#!/usr/bin/env python
# coding: utf-8

# ## Collecting Data From CURE Deliverable 3

# #### Creating a table from excel

# In[2]:


import pandas as pd
from pandas import Series, DataFrame

df = DataFrame()
df_from_excel = pd.read_excel(r'C:\Users\habib\Downloads\Deliverable_3_.xlsx')
dfsorted = df_from_excel.sort_values ('continents', ignore_index=True)
f0, f1, f2 = dfsorted.continents.value_counts()
dfsorted


# ## Histogram and Scatter Plots from CURE Deliverable 3

# In[5]:


import matplotlib.pyplot as plt
import numpy as np
fig, axs = plt.subplots( figsize = (12,3))

axs1 = plt.subplot2grid ( shape = (1, 3), loc = (0,0))
axs2 = plt.subplot2grid ( shape = (1, 3), loc = (0,1))
axs3 = plt.subplot2grid ( shape = (1, 3), loc = (0,2))
plt.tight_layout()

axs1.hist (dfsorted.iloc[0:f1, 2], edgecolor = 'b', fc = 'none', label = 'Asia')
axs1.hist (dfsorted.iloc[f1:f0+f1, 2], edgecolor = 'r', fc = 'none', label = 'Europe')
axs1.hist (dfsorted.iloc[f0+f1:f0+f1+f2, 2], edgecolor = 'g', fc = 'none', label = 'NA')
axs1.set_xlabel ('Population')
axs1.set_ylabel ('Frequency')
axs1.legend()

axs2.scatter (dfsorted.iloc[0:f0, 3], dfsorted.iloc[0:f0, 2], label = 'Asia')
axs2.scatter (dfsorted.iloc[f0:f0+f1, 3], dfsorted.iloc[f0:f0+f1, 2], label = 'Europe')
axs2.scatter (dfsorted.iloc[f0+f1:f0+f1+f2, 3], dfsorted.iloc[f0+f1:f0+f1+f2, 2], label = 'NA')
axs2.set_xlabel ('Land Size (KM^2)')
axs2.set_ylabel ('Population')
axs2.legend()

axs3.scatter (dfsorted.iloc[0:f0, 4], dfsorted.iloc[0:f0, 2], label = 'Asia')
axs3.scatter (dfsorted.iloc[f0:f0+f1, 4], dfsorted.iloc[f0:f0+f1, 2], label = 'Europe')
axs3.scatter (dfsorted.iloc[f0+f1:f0+f1+f2, 4], dfsorted.iloc[f0+f1:f0+f1+f2, 2], label = 'NA')
axs3.set_xlabel ('Quality of Life Index')
axs3.set_ylabel ('Population')
axs3.legend()


# ## CURE Deliverable 4:
# ## Task 1_Develop a machine learning model
# ### Part I: Divide the data into training and test set
# 

# In[63]:


df = dfsorted.iloc[0:100,[0,2,3,4]]
dfX = df.drop (columns =['continents'])
sy = df.continents
from sklearn.model_selection import train_test_split
dfX_train, dfX_test, sy_train, sy_test = train_test_split(dfX,sy)
dfX_train, dfX_test, sy_train, sy_test


# ### Part II: Data Preprocessing
# #### Preprocessing non-numerical classes with LabelEncoder()

# In[64]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le = le.fit(sy_train)
le.classes_


# In[65]:


y_train = le.transform(sy_train)
sy_train, y_train


# #### Preprocessing attributes with MinMaxScaler()

# In[ ]:


dfX_train.to_numpy()


# In[ ]:


n1 = preprocessing.MinMaxScaler()
n1 = n1.fit(dfX_train.to_numpy())
X_train = n1.transform (dfX_train.to_numpy())
X_train


# ## Task 2_Doing some analysis to find best value of k
# ### 2(a) Iterating through the range of samples (using a for loop) in order to get all possible accuracy values corresponding to different k values

# In[68]:


from sklearn.neighbors import KNeighborsClassifier
scores = DataFrame()
knn = KNeighborsClassifier(n_neighbors=1)
knn = knn.fit(X_train,y_train)
y_test = le.transform(sy_test.to_numpy())
X_test = n1.transform(dfX_test.to_numpy())
scores['k_value'] = []
scores['testing_score'] = []
scores['training_score'] = []

for i in range(1,76):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn = knn.fit(X_train,y_train)
    y_test = le.transform(sy_test.to_numpy())
    X_test = n1.transform(dfX_test.to_numpy())
    scores.loc[+i] = [i,knn.score(X_test,y_test),knn.score(X_train,y_train)]
scores


# ### Scatter plot modelling the accuracy for different k values with respect to their training/tesing scores

# In[69]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots()

ax.scatter(scores.k_value, scores.testing_score, label = 'testing score')
ax.scatter(scores.k_value, scores.training_score, label = 'training score')

plt.xlabel('k-value')
plt.ylabel('Accuracy')
plt.title('Accuracy scores using MinMaxScaler()')
plt.legend()
plt.show()


# ### 2(b) Repeating Task 2(a) by using StandardScaler() instead of MinMaxScaler()

# In[71]:


n1 = preprocessing.StandardScaler()
n1 = n1.fit(dfX_train.to_numpy())
X_train = n1.transform (dfX_train.to_numpy())
from sklearn.neighbors import KNeighborsClassifier
scores = DataFrame()
knn = KNeighborsClassifier(n_neighbors=1)
knn = knn.fit(X_train,y_train)
y_test = le.transform(sy_test.to_numpy())
X_test = n1.transform(dfX_test.to_numpy())
scores['k_value'] = []
scores['testing_score'] = []
scores['training_score'] = []

for i in range(1,76):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn = knn.fit(X_train,y_train)
    y_test = le.transform(sy_test.to_numpy())
    X_test = n1.transform(dfX_test.to_numpy())
    scores.loc[+i] = [i,knn.score(X_test,y_test),knn.score(X_train,y_train)]
    
import matplotlib.pyplot as plt
fig, ax = plt.subplots()

ax.scatter(scores.k_value, scores.testing_score, label = 'testing score')
ax.scatter(scores.k_value, scores.training_score, label = 'training score')

plt.xlabel('k-value')
plt.ylabel('Accuracy')
plt.title('Accuracy scores using StandardScaler()')
plt.legend()
plt.show()


# ## Task 3_Finding best value of k
# #### (a) Our group choose the MinMax as the best scaler for our model. Using this scaler, our group selected the best value of k to be 19
# 

# In[82]:


from sklearn.metrics import confusion_matrix
y_pred = knn.predict(X_test)
confusion_matrix(y_test, y_pred)


# In[83]:


from sklearn.utils.multiclass import unique_labels
unique_labels(y_test)


# #### Note: In the confusion matrix, 0 corresponds to Asia, 1 corresponds to Europe, and 2 corresponds to North America

# In[85]:


def plot(y_true, y_pred):
    labels = unique_labels(y_test)
    columns = [f'Predicted{label}' for label in labels]
    index = [f'Actual{label}' for label in labels]
    table = pd.DataFrame(confusion_matrix(y_true, y_pred), columns=columns, index=index)
    
    return table
plot(y_test, y_pred)


# ## Task 4_Apply your model
# #### We classified a new instance Warsaw which is a city in Poland, Europe. Warsaw has a population of 1.765 million people, a land size of 517.2 km^2, and a quality of life index of 119.80. Our model predicted the right class for this instance which was class 1 corresponding to Europe.

# In[89]:


new_example_rawdata = np.array([[1765000, 517.2, 119.80]])
X_new = n1.transform(new_example_rawdata)
X_new


# In[90]:


knn.predict(X_new)


# In[91]:


le.inverse_transform(knn.predict(X_new))


# In[ ]:




