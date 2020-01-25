#!/usr/bin/env python
# coding: utf-8

# # CS 514 Applied AI
# 
# ## ASHRAE - Great Energy Prediction 3
# 
# https://www.kaggle.com/c/ashrae-energy-prediction/overview

# # 1. Importing all the Libraries

# In[1]:


#Ignoring all System warnings
import warnings
warnings.filterwarnings('ignore')

#Importing Garbage Collector and the system call to os
import gc
import os

#Importing all the Scientific Libraries
import pandas as pd
import numpy as np
import scipy as sc

#Importing all the plotting libraries
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder


# In[2]:


import random
random.seed(0)


# # 2. Reading the Data

# In[3]:


os.getcwd()


# In[4]:


print(os.listdir())


# In[5]:


train_df = pd.read_csv("train.csv")

building_df = pd.read_csv("building_metadata.csv")

weather_train_df = pd.read_csv("weather_train.csv")

weather_test_df = pd.read_csv("weather_test.csv")

test_df = pd.read_csv("test.csv")


# ## Know more about the data

# Applying df.describe() functions to every dataframe to know about the stats of the data

# In[6]:


print("Describing the Train data\n",train_df.describe())
print("\n")
print("Describing the building data\n",building_df.describe())
print("\n")
print("Describing the weather train data\n",weather_train_df.describe())
print("\n")
print("Describing the weather test data\n",weather_test_df.describe())
print("\n")
print("Describing the Test data\n",test_df.describe())
print("\n")


# Applying df.shape() function to every df to know the number of rows and columns in the df.
# 
# 
# Here the result will be (x,y) where x is the number of row and y is the number of column

# In[7]:


print("Shape of Train data: ",train_df.shape)
print("Shape of building data: ",building_df.shape)
print("Shape of weather train data: ",weather_train_df.shape)
print("Shape of weather test data: ",weather_test_df.shape)
print("Shape of Test data: ",test_df.shape)


# Applying df.head() to display rows of the dataframe

# In[8]:


train_df.head()


# In[9]:


building_df.head()


# Here we see primary_use is a categorical variable which can be encoded using LabelEncoder().
# 
# Below we will look at all the unique categories of the primary_use column in the building_df.

# In[10]:


building_df['primary_use'].unique().tolist()


# Label Encoding refers to converting the labels into numeric form so as to convert it into the machine-readable form
# 
# Example :
# <br>
# Suppose we have a column Height in some dataset.
# 
# Height = ["small","medium","Tall"]
# 
# Then after encoding the Height will become as: Height = [0,1,2]

# In[11]:


le = LabelEncoder()
building_df['primary_use'] = le.fit_transform(building_df['primary_use'])


# In[12]:


building_df.head()


# In[13]:


weather_train_df.head()


# In[14]:


weather_test_df.head()


# In[15]:


test_df.head()


# # 3. Reducing the Memory Usage

# Some features take up more memory space than they should, and since there is too much data, this is critical step to reduce the memory use by the data.

# In[16]:


# we can count the actual memory usage using the following command
print("Memory used by train_df:")
train_df.info(memory_usage='deep')
print("\n Memory used by building_df:")
building_df.info(memory_usage='deep')
print("\n Memory used by weather_train_df:")
weather_train_df.info(memory_usage='deep')
print("\n Memory used by weather_test_df:")
weather_test_df.info(memory_usage='deep')
print("\n Memory used by test_df:")
test_df.info(memory_usage='deep')


# In[17]:


# we can check how much space each column is actually taking
# the numbers are in bytes, not kilobytes
train_df.memory_usage('deep')


# Below is the Table showing memory usage by different datatypes

# ![](D:\KaggleProjects\ASHRAE\ashrae-energy-prediction\table.png)

# ![memory usage](table.png)

# ## Function to define reduce memory usage

# In[18]:


def reduce_mem_usage(df):
    start_mem_usage = df.memory_usage().sum() / 1024**2 #Convert bytes to megabytes(MB)
    print("Memories usage of the dataframe is :",start_mem_usage,"MB")
    list_na = [] ## Keeps track of columns that have missing values filled in. 
    for col in df.columns:
        if df[col].dtype != object:#Excluding string
            # make variables for Int, max and min
            IsInt = False
            mx = df[col].max()
            mn = df[col].min()
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(df[col]).all(): 
                list_na.append(col)
                df[col].fillna(mn-1,inplace=True) 
            
            # test if column can be converted to an integer
            asint = df[col].fillna(0).astype(np.int64)
            result = (df[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True
                
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)   
                        
             # Make float datatypes 32 bit
            else:
                df[col] = df[col].astype(np.float32)
                
            # Print new column type
            print("dtype after: ",df[col].dtype)
            print("******************************")
            
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usage,"% of the initial size")
    return df, list_na
        


# In[19]:


train_df, NAlist = reduce_mem_usage(train_df)
print("_________________")
print("")
print("Warning: the following columns have missing values filled with 'df['column_name'].min() -1': ")
print("_________________")
print("")
print(NAlist)


# In[20]:


test_df, NAlist = reduce_mem_usage(test_df)
print("_________________")
print("")
print("Warning: the following columns have missing values filled with 'df['column_name'].min() -1': ")
print("_________________")
print("")
print(NAlist)


# In[21]:


building_df, NAlist = reduce_mem_usage(building_df)
print("_________________")
print("")
print("Warning: the following columns have missing values filled with 'df['column_name'].min() -1': ")
print("_________________")
print("")
print(NAlist)


# In[22]:


building_df.isnull().any().sum()


# In[23]:


weather_train_df, NAlist = reduce_mem_usage(weather_train_df)
print("_________________")
print("")
print("Warning: the following columns have missing values filled with 'df['column_name'].min() -1': ")
print("_________________")
print("")
print(NAlist)


# In[24]:


weather_train_df.isnull().any().sum()


# In[25]:


weather_test_df, NAlist = reduce_mem_usage(weather_test_df)
print("_________________")
print("")
print("Warning: the following columns have missing values filled with 'df['column_name'].min() -1': ")
print("_________________")
print("")
print(NAlist)


# In[26]:


weather_test_df.isnull().any().sum()


# # 4. Merging the data 

# In[27]:


def merge_data(train,building,weather,test=False):
    """Merging building and weather data with train and test data"""
    train = train.merge(building_df,on="building_id",how ="left")
    train = train.merge(weather_train_df,on=["site_id","timestamp"],how="left")
    
    
    train['timestamp'] = pd.to_datetime(train['timestamp'],format="%Y-%m-%d %H:%M:%S")
    train['square_feet'] = np.log1p(train['square_feet'])#np.log1p is used so that smallest value aren't ignored
    
    if not test:
        train.sort_values('timestamp',inplace=True)
        train.reset_index(drop=True,inplace=True)
        
        
    gc.collect()
    
    holidays = ["2016-01-01", "2016-01-18", "2016-02-15", "2016-05-30", "2016-07-04",
                "2016-09-05", "2016-10-10", "2016-11-11", "2016-11-24", "2016-12-26",
                "2017-01-01", "2017-01-16", "2017-02-20", "2017-05-29", "2017-07-04",
                "2017-09-04", "2017-10-09", "2017-11-10", "2017-11-23", "2017-12-25",
                "2018-01-01", "2018-01-15", "2018-02-19", "2018-05-28", "2018-07-04",
                "2018-09-03", "2018-10-08", "2018-11-12", "2018-11-22", "2018-12-25",
                "2019-01-01"]
        
    train["hour"] = train.timestamp.dt.hour
    train["weekday"] = train.timestamp.dt.weekday
    train["is_holiday"] = (train.timestamp.dt.date.astype("str").isin(holidays)).astype(int)
    
    train.drop("timestamp",axis=1,inplace=True)
    
    if test:
        row_ids = train.row_id
        train.drop("row_id", axis=1, inplace=True)
        return train, row_ids
    
    else:
        y = np.log1p(train.meter_reading)
        train.drop("meter_reading", axis=1, inplace=True)
        return train, y
    


# In[28]:


train_X,train_y = merge_data(train_df,building_df,weather_train_df)
#gc.collect()


# In[29]:


test_x,row_ids = merge_data(test_df,building_df,weather_test_df,test=True)


# In[30]:


gc.collect()


# In[31]:


np.isnan(train_X).sum()


# In[32]:


train_X["air_temperature"].fillna(0,inplace=True)


# In[33]:


train_X["cloud_coverage"].fillna(0,inplace=True)


# In[34]:


train_X["dew_temperature"].fillna(0,inplace=True)


# In[35]:


train_X["precip_depth_1_hr"].fillna(0,inplace=True)


# In[36]:


train_X["sea_level_pressure"].fillna(0,inplace=True)


# In[37]:


train_X["wind_direction"].fillna(0,inplace=True)


# In[38]:


train_X["wind_speed"].fillna(0,inplace=True)


# In[39]:


train_X.isnull().any().sum()


# In[40]:


test_x.shape


# In[41]:


test_x.isnull().any().sum()


# In[42]:


np.isnan(test_x).sum()


# In[43]:


test_x["air_temperature"].fillna(0,inplace=True)


# In[44]:


test_x["cloud_coverage"].fillna(0,inplace=True)


# In[45]:


test_x["dew_temperature"].fillna(0,inplace=True)


# In[46]:


test_x["precip_depth_1_hr"].fillna(0,inplace=True)


# In[47]:


test_x["sea_level_pressure"].fillna(0,inplace=True)


# In[48]:


test_x["wind_direction"].fillna(0,inplace=True)


# In[49]:


test_x["wind_speed"].fillna(0,inplace=True)


# In[50]:


test_x.isnull().sum()


# # 5. Train and Test Split

# In[51]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


# In[52]:


from sklearn.model_selection import train_test_split


# Splitting the data into train and validation set to know which model will perform better

# In[53]:


X_train, X_valid, y_train, y_valid = train_test_split(train_X,train_y,test_size=0.50,shuffle = True)


# # Linear Regression Model

# In[54]:


lr = LinearRegression()

lr.fit(X_train,y_train)
pred_reg = lr.predict(X_valid)


# In[55]:


pred_reg=pred_reg.reshape(-1,1)


# In[56]:


pred_reg


# In[57]:


y_valid = y_valid.values.reshape(-1,1)


# In[58]:


y_valid


# In[59]:


pred_reg.shape


# In[60]:


y_valid.shape


# In[ ]:





# # Function which computes root mean squared log error

# In[62]:


def root_mean_squared_log_error(real, predicted):
    sum=0.0
    for x in range(len(predicted)):
        if predicted[x]<0 or real[x]<0: # check for negative values
            continue
        p = np.log1p(predicted[x]+1)
        r = np.log1p(real[x]+1)
        sum = sum + (p - r)**2
    return (sum/len(predicted))**0.5


# In[63]:


root_mean_squared_log_error(y_valid,pred_reg)


# # Calculating mean square error

# In[64]:


from math import sqrt
rms = sqrt(mean_squared_error(y_valid,pred_reg))


# In[65]:


print(rms)


# In[ ]:





# # Decision Tree Regressor Model

# In[66]:


dt = DecisionTreeRegressor(criterion="mse",min_samples_leaf=5)
dt.fit(X_train,y_train)
pred_dt = dt.predict(X_valid)


# In[67]:


pred_dt=pred_dt.reshape(-1,1)


# In[68]:


rms_d = sqrt(mean_squared_error(y_valid,pred_dt))
print(rms_d)


# In[69]:


root_mean_squared_log_error(y_valid,pred_dt)


# In[ ]:





# # SGDRegressor Model

# In[70]:


from sklearn import linear_model


# In[71]:


sgd_reg = linear_model.SGDRegressor(max_iter=1000,tol=1e-3)
sgd_reg.fit(X_train,y_train)
pred_sgd = sgd_reg.predict(X_valid)


# In[72]:


pred_sgd = pred_sgd.reshape(-1,1)


# In[73]:


rms_sgd = sqrt(mean_squared_error(y_valid,pred_sgd))
print(rms_sgd)


# In[74]:


root_mean_squared_log_error(y_valid,pred_sgd)


# # Deleting all the unused variable to clear the memory 

# In[75]:


del train_df,building_df,weather_train_df,weather_test_df,test_df


# In[76]:


del lr


# In[77]:


del dt


# In[78]:


del X_train, X_valid, y_train, y_valid 


# In[79]:


del sgd_reg 


# In[80]:


gc.collect()


# # 6. Model Preparation

# Finally Selecting the Decision Tree Regressor since it has low root mean square error

# In[81]:


regression_model = DecisionTreeRegressor(criterion="mse",min_samples_leaf=5)


# In[82]:


regression_model.fit(train_X,train_y)


# In[83]:


predicted = regression_model.predict(test_x)


# Making a DataFrame of the predicted results and finally saving it to ashrae_submit.csv

# In[87]:


submission_df = pd.DataFrame(zip(row_ids,predicted),columns = ['row_id','meter_reading'])


# In[90]:


submission_df.shape


# In[91]:


submission_df.to_csv("ashrae_submit.csv", index=False)


# In[ ]:




