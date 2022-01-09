#!/usr/bin/env python
# coding: utf-8

# In[14]:


from sklearn.datasets import load_diabetes
diabetesdata = load_diabetes()
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
lrmodel = LinearRegression()
diabetesdata.keys()
X_vals = diabetesdata['data']
Y_vals = diabetesdata['target']
X_vals.shape #(442, 10) means 442 instances and 10 features per instance
Y_vals.shape #442 labels (1 per instance)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X_vals, Y_vals, test_size = 0.2)
lrmodel.fit(X_train, Y_train)
vals = lrmodel.predict(X_test)
from sklearn.metrics import mean_squared_error
mean_squared_error(Y_test, vals)
# we get a mean squared error of 2,831 however lets see what happens when we normalize the data first


# In[36]:


from sklearn.preprocessing import StandardScaler
df = pd.DataFrame(np.c_[X_vals, Y_vals])
scaler = StandardScaler()
diabetesdatascaled = scaler.fit_transform(df)
dfscaled = pd.DataFrame(diabetesdatascaled)
yvalsscaled = dfscaled[10]
xvalsscaled = dfscaled.drop([10], axis = "columns")
from sklearn.model_selection import train_test_split
X_sTrain, X_sTest, Y_sTrain, Y_sTest = train_test_split(xvalsscaled, yvalsscaled, test_size = 0.2)
from sklearn.linear_model import LinearRegression
lrmodel = LinearRegression()
lrmodel.fit(X_sTrain,Y_sTrain)
predictions = lrmodel.predict(X_sTest)
from sklearn.metrics import mean_squared_error
mean_squared_error(Y_sTest, predictions) #The error amount we got here with scaled data was .48 with a range of 3.5 within the 
#labels


# In[42]:


from sklearn.metrics import mean_absolute_error
mean_absolute_error(Y_sTest, predictions)
mean_squared_error(Y_sTest, predictions)


# In[73]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
higherdimreg =LinearRegression(normalize = True)
higherdimreg.fit(X_train, Y_train)
vals = higherdimreg.predict(X_test)
mean_squared_error(Y_test, vals)
# Y_test


# In[ ]:




