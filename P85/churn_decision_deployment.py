#!/usr/bin/env python
# coding: utf-8

# ## 1. Importing necessary libraries

# In[54]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pycountry as pc
import matplotlib.ticker as mtick
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# ## 2. import data

# In[55]:


booking_data = pd.read_csv('Bookings.csv')


# ## Data preprocessing

# In[56]:


df=booking_data.copy()


# In[57]:


# replacing null values of agent with 0 
df[['agent','company']]=df[['agent','company']].fillna(0.0)


# In[58]:


# replacing country and children values with mean and mode imputation
df['country'].fillna(booking_data.country.mode().to_string(), inplace = True)


df['children'].fillna(round(booking_data.children.mean()), inplace = True)


# In[59]:


# treating undefined values in meal column
df['meal'].replace(to_replace='Undefined',value='BB',inplace=True)


# In[60]:


# treating undefined values in distribution_channel column
df['distribution_channel'].replace(to_replace='Undefined',value='TA/TO',inplace=True)


# In[61]:


# treating undefined values in market_segment cdolumn 
df['market_segment'].replace(to_replace='Undefined',value='Online TA',inplace=True)


# In[62]:


# deleting the duplicate values
df.drop_duplicates(inplace=True)


# In[63]:


df[(df.adults+df.babies+df.children)==0]


# In[64]:


df = df.drop(df[(df.adults+df.babies+df.children)==0].index)


# In[65]:


# converting data types
df[['children', 'company', 'agent']] = df[['children', 'company', 'agent']].astype('int64')


# ## Feature selection and feature engineering
# #### lets try without splitting the hotels and try to apply decision tree or random forest 

# In[66]:


## copy the dataframe
df_subset=df.copy()


# In[67]:


# make a new column: which says 1 if the customer got the same room he reserved for by 
df_subset['Room']= 0
df_subset.loc[df_subset['reserved_room_type']==df_subset['assigned_room_type'],'Room']=1


# In[68]:


# making a new column which contains 1 if the previous cancellations
# are more than the previous not cancelled 
df_subset['net_cancelled']=0
df_subset.loc[df_subset['previous_cancellations']>df_subset['previous_bookings_not_canceled'],'net_cancelled']=1


# In[69]:


df_subset=df_subset.drop(['arrival_date_year','arrival_date_month','arrival_date_week_number','arrival_date_day_of_month','previous_cancellations','previous_bookings_not_canceled','reserved_room_type','assigned_room_type','reservation_status','reservation_status_date'],axis=1)


# ### Applying Label encodig

# In[70]:


cols=['hotel','meal','country','market_segment','distribution_channel','deposit_type','customer_type']
df_subset[cols]=df_subset[cols].apply(LabelEncoder().fit_transform)


# ### Train test split

# In[71]:


X=df_subset.drop('is_canceled', axis=1)
y=df_subset['is_canceled']


# In[72]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# In[73]:


clf=DecisionTreeClassifier()
clf.fit(X_train,y_train)


# In[74]:


train_score= clf.score(X_train, y_train)
test_score = clf.score(X_test, y_test)


# In[75]:


print('Training accuracy of decision tree model is: ',train_score)


# In[76]:


prediction = clf.predict(X_train.iloc[15].values.reshape(1,-1))

actual_value = y_train.iloc[15]

print(f'Predicted Value \t: {prediction[0]}')
print(f'Actual Value\t\t: {actual_value}')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




