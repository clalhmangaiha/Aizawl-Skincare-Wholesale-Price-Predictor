#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


X = pd.read_csv('some.csv')
# X_test = pd.read_csv('test.csv')


# In[4]:


X.head()


# In[5]:


#Target values neilo kan remove dawn (drop) Rows bik ah
X.dropna(axis=0,subset=['price1'],inplace=True)


# In[7]:


#Target kan assign ang
y = X.price1


# In[8]:


#Separate target leh predictors
X.drop(['price1'],axis=1,inplace=True)


# In[9]:


X.head()


# In[10]:


#breaking off validation and testing data
from sklearn.model_selection import train_test_split

x_train,x_valid,y_train,y_valid = train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=1)


# In[30]:


y_valid


# In[11]:



#Numerical columns
numerical_columns = [val for val in x_train.columns
                    if x_train[val].dtype in ['int64','float64']]


# In[12]:


numerical_columns


# In[13]:


#Categorical Columns
categorical_columns = [val for val in x_train.columns
                      if x_train[val].nunique()<10 and
                      x_train[val].dtype == 'object']


# In[14]:


categorical_columns


# In[15]:


#Combine numerical & categorical columns
selected_columns = numerical_columns + categorical_columns


# In[17]:


x_train = x_train[selected_columns].copy()
x_valid = x_valid[selected_columns].copy()
# X_test = X_test[selected_columns].copy()


# In[18]:


#Preprocessing

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


# In[19]:


#numerical column transformer
numerical_transformer = SimpleImputer(strategy='constant')


# In[20]:


#categorical column transformeabsr
categorical_transformer = Pipeline(steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('onehot',OneHotEncoder(handle_unknown='ignore'))
])


# In[21]:


from sklearn.compose import ColumnTransformer

preprocessing = ColumnTransformer(
                transformers=[
                    ('num',numerical_transformer,numerical_columns),
                    ('cat',categorical_transformer,categorical_columns),
                ])


# In[22]:


#model define RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=200,random_state=1)


# In[23]:


#Preprocessing leh model

p = Pipeline(steps=[
    ('preprocessing',preprocessing),
    ('model',model),
])


# In[24]:


p.fit(x_train,y_train)


# In[25]:


p.predict(x_valid)


# In[26]:


from sklearn.metrics import mean_absolute_error

score = mean_absolute_error(p.predict(x_valid),y_valid)


# In[27]:


score


# In[37]:


some_predict = pd.read_csv('some_predict.csv')
some_predict = some_predict[selected_columns].copy()
some_predict


# In[38]:


p.predict(some_predict)


# In[ ]:




