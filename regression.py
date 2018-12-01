
# coding: utf-8

#     On commence par importer les librairies dont on a besoin.

# In[1]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#     On importe nos données a partir du fichier csv 

# In[2]:


# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')  ###
dataset


# In[3]:


X = dataset.iloc[:, :-1].values      ###
y = dataset.iloc[:, 4].values        ###  


#     on trouve un moyen pour regler le probleme des données manquantes

# In[4]:


# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)   
imputer = imputer.fit(X[:, 0:3])                      
X[:, 0:3] = imputer.transform(X[:, 0:3])


#     on transforme les données logiques en une forme numerique afain de pouvoir les exploiter.

# In[5]:


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])     ###
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
# Avoiding the Dummy Variable Trap
X = X[:, 1:]                            ###


#     on divise nos données en deux :
#             ensemble pour l'entrainement.
#             ensemble pour le test.

# In[6]:


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)    ### 


#     creation du regresseur.

# In[7]:


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


#     prediction des resultats.

# In[8]:


# Predicting the Test set results
y_pred = regressor.predict(X_test)


# In[9]:


pd.DataFrame(data=[y_pred,y_test])


# In[10]:


from sklearn.metrics import mean_absolute_error
error =mean_absolute_error(y_test, y_pred, sample_weight=None)     ###
print('error of prdiction: '+str(int(error))+'$')


#     On refait la meme chose mais uniquement avec la colone R&D Spendings.

# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X[:,2].reshape(-1, 1), y, test_size = 0.2, random_state = 0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
pd.DataFrame(data=[y_pred,y_test])


# In[12]:


error =mean_absolute_error(y_test, y_pred, sample_weight=None)
print('error of prdiction: '+str(int(error))+'$')

