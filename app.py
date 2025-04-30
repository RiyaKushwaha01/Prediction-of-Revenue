#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyodbc

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Moudles related to feature selection
from sklearn.feature_selection import RFE, SelectKBest , f_regression


# In[2]:


df = pd.read_csv(r"C:\Users\ADMIN\OneDrive\Desktop\Internship\Project 5 (Oil and Gas Industry )\Streamlit Files\Data\Natural_Resources_Revenue.csv")


# In[5]:


df.isnull().sum()


# In[4]:


df.head()


# In[5]:


df = df[df['Product'].notna()]


# In[6]:


df.info()


# In[7]:


df.shape


# In[8]:


df.isnull().sum()


# In[9]:


df = df.drop(columns=['Offshore Region','Calendar Year','FIPS Code'])


# In[10]:


df.head()


# In[11]:


df.head()


# In[13]:


df.info()


# In[14]:


df = df.apply(lambda col: col.fillna(col.mode()[0]) if col.dtype == 'object' else col.fillna(col.median()))


# In[15]:


df.info()


# In[16]:


X=df[df.columns.difference(['Revenue'])]
y=df.Revenue


# In[17]:


X_train, X_val, y_train, y_val=train_test_split(X, y, test_size=0.3, random_state=37)


# In[18]:


df.head()


# In[35]:


from catboost import CatBoostRegressor

# Define categorical feature columns
cat_features = ['Land Class', 'Land Category', 'State', 'Revenue Type', 'Mineral Lease Type', 'Commodity','County','Product']

# Initialize CatBoostRegressor
model = CatBoostRegressor(iterations=100, loss_function='RMSE', random_state=42, verbose=0)

# Fit the model
model.fit(X_train, y_train, cat_features=cat_features)

# Predict
y_pred = model.predict(X_val)

# Evaluate
print("MAE:", mean_absolute_error(y_val, y_pred))
print("MAPE:", mean_absolute_percentage_error(y_val, y_pred))
print("RMSE:", mean_squared_error(y_val, y_pred, squared=False))
print("R2 Score:", r2_score(y_val, y_pred))


# In[378]:


df.info()


# In[37]:


from catboost import CatBoostRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score

# Define categorical feature columns
cat_features = ['Land Class', 'Land Category', 'State', 'Revenue Type', 'Mineral Lease Type', 'Commodity', 'County', 'Product']

# Initialize the base model
model = CatBoostRegressor(loss_function='RMSE', random_state=42, verbose=0)

# Define hyperparameter space
param_dist = {
    'iterations': [100, 200, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'depth': [4, 6, 8, 10],
    'l2_leaf_reg': [1, 3, 5, 7, 9],
    'bagging_temperature': [0, 1, 2, 5],
    'border_count': [32, 64, 128]
}

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=20,   # number of different combinations to try
    cv=3,        # 3-fold cross-validation
    verbose=1,
    random_state=42,
    n_jobs=-1    # use all available cores
)

# Fit RandomizedSearchCV
random_search.fit(X_train, y_train, cat_features=cat_features)

# Best model
best_model = random_search.best_estimator_

# Predict
y_pred = best_model.predict(X_val)

# Evaluate
print("MAE:", mean_absolute_error(y_val, y_pred))
print("MAPE:", mean_absolute_percentage_error(y_val, y_pred))
print("RMSE:", mean_squared_error(y_val, y_pred, squared=False))
print("R2 Score:", r2_score(y_val, y_pred))

# Optional: print best parameters
print("Best Parameters:", random_search.best_params_)


# In[ ]:


import pickle

# Example: Save a model or DataFrame
with open('Model.pkl', 'wb') as file:
    pickle.dump(your_object, file)

