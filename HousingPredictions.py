#!/usr/bin/env python
# coding: utf-8

# In[165]:


import numpy as np
import pandas as  pd 
traindata = pd.read_csv("train.csv")
traindata
targetvals = traindata[['SalePrice']]


# In[166]:


#Before we start feature engineering, im gonna split the target column from the rest of the columns


# In[167]:


trainvals = traindata.drop("SalePrice", axis = "columns")


# In[168]:


trainvals #Now that we dont have the target column, we can start editing features and removing what isnt important from 
#intuition
#What im thinking is because we have so many variables, we can run this through a pipeline to fill in any missing vals
#that are numeric.
trainvals.info()
#Upon inspection of this data, we notice that column "PoolQC" has 7 non nulls which means it has 1453 null values in its column
#So we can drop this because it will really not matter, additionally if we have the misc feature values, the misc features
#themselves wont really matter at all. 
#What we can do instead of giving a damn about the fences column is to keep track of the indices in relation to fences, and
#then when we make predictions on these specific indices we add 1342$ to these predictions as after doing some research
#i found that fences in iowa city iowa increase house cost by 1342 on average. 
# We can also drop alley because it wont have much of an affect as it only is involved with 91 out of 1460 instances.
fencebooleans = trainvals['Fence'].notnull()
fenceindices = np.where(fencebooleans)[0]
fenceindices #Ok now we have fence indices, remembering 1342$ for every prediction that satisfies these indices
#Dropping.. PoolQC, Fence, Alley MiscFeature #Since we have 1460 MiscVal we dont need MiscFeature


# In[169]:


#Now we can use these indices in column transformer to convert all of the missing vals in numeric data to have the mean of 
#thoserespective coluns. This is great because we have so many instances in most of those columns that are filled in that
#the mean should give a good representation of what to expect.
#Also while were at it I just noticed that these values are very much spread out so I will consider normalizing a bit later
#to see if it results in any steady improvements.
#We will use ordinal encoder for the object columns
trainvals = trainvals.drop(columns = ['PoolQC','Fence', 'Alley', 'MiscFeature'])
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numcol = trainvals.select_dtypes(include = numerics).columns
objectcol = trainvals.select_dtypes(include = 'object').columns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
#Almost forgot to add in a replacement for nan values in object columns, what I can do is make the valus we fill into nan in 
#object columns equal to the mode.
ct = ColumnTransformer([("Numeric Fixers", SimpleImputer(missing_values = np.nan, strategy = "mean"), numcol),
                        ("ONANFILL",SimpleImputer(missing_values = np.nan, strategy = "most_frequent"), objectcol)])
transformedvals = ct.fit_transform(trainvals)
transformedvals


# In[170]:


#In our first go all we did was fill in the numeric columns if they had any missing values with the mean in their respective
#columns, and then we dropped a couple features and kept track of the fence features indices to add an extra 1300$ to the
#prediction. We will see the results we get with this and we if we notice poor performance we will do even more inspection of 
#features and what they mean.
transformedvals.shape


# In[171]:


transformeddf = pd.DataFrame(transformedvals, columns = trainvals.columns)
transformeddf = transformeddf.convert_dtypes()
transformeddf.info()


# In[172]:


#Now lets make sure we are seeing right and double check to see if there any nan values 
#Now that we have completed this task, lets employ ordinal encoder on all string type of values.
Stringindices = transformeddf.select_dtypes(include = 'string').columns
Stringindices
from sklearn.preprocessing import OrdinalEncoder
transformeddf[Stringindices] = OrdinalEncoder().fit_transform(transformeddf[Stringindices])
transformeddf.info() #Ok now we did ordinal encoder


# In[210]:


#Now that we have a basic amount of pre processing complete we can try different models. Remember we can still try and normalize
#but before lets split this stuff into training and validation set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(transformeddf, targetvals, test_size = 0.2)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
rfrbasic = RandomForestRegressor(n_estimators = 200)
rfrbasic.fit(X_train, y_train)
predictions = rfrbasic.predict(X_test)
mean_absolute_error(y_test, predictions)


# In[197]:


#Lets look at the mean value of a house and see how much of a difference 17k really means relative to the housing cost
targetvals.mean() #We get a value of 180,921 as the mean this means that
percentoff = 17224.43544520548/180921.19589
print(percentoff) #We are off about 9.5% on average each prediction. Which is really not too bad! :D
#lets see if we can improve this score on average with some hyperparameter tuning.


# In[198]:


from sklearn.model_selection import GridSearchCV
rfrparams = {
    'n_estimators':[50,100,200,250,300,400],
    'criterion':['squared_error', 'absolute_error', 'poisson'],
    'min_samples_leaf':[10,20,30]
}
from sklearn.ensemble import RandomForestRegressor
rfrtuningbby = GridSearchCV(RandomForestRegressor(), rfrparams, cv = 5)
rfrtuningbby.fit(transformeddf, targetvals)
rfrtdf = pd.DataFrame(rfrtuningbby.cv_results_)


# In[199]:


rfrtuningbby.best_score_


# In[201]:


testmodel = rfrtuningbby.best_estimator_


# In[204]:


testmodel.fit(X_train, y_train)
predict = testmodel.predict(X_test)
mean_absolute_error(y_test, predict) #So after testing the best model rf model with grid search, it turns out best rfr model 
#is the basic rfr model except with 300 trees.


# In[209]:


mean_absolute_error(y_test, predictions) #Approx 8,000$ more accurate than gridsearchcv model.


# In[211]:


testdata = pd.read_csv("test.csv")


# In[213]:


testdata.info()


# In[215]:


#Now we are working on setting up the kaggle predictions
testct = ColumnTransformer([("Numeric Fixers", SimpleImputer(missing_values = np.nan, strategy = "mean"), numcol),
                        ("ONANFILL",SimpleImputer(missing_values = np.nan, strategy = "most_frequent"), objectcol)])
transformedtestvals = testct.fit_transform(testdata)
transformedtestdf = pd.DataFrame(transformedtestvals, columns = trainvals.columns)
transformedtestdf = transformedtestdf.convert_dtypes()
Stringindices = transformedtestdf.select_dtypes(include = 'string').columns
Stringindices
from sklearn.preprocessing import OrdinalEncoder
transformedtestdf[Stringindices] = OrdinalEncoder().fit_transform(transformedtestdf[Stringindices])
finaloutput = rfrbasic.predict(transformedtestdf)
finaloutput


# In[223]:


finaldf = pd.DataFrame(np.c_[transformedtestdf[['Id']], finaloutput], columns = ['Id', 'SalePrice'])


# In[224]:


finaldf


# In[232]:


finaldf.to_excel(r"C:\Users\tun84049\Desktop\KaggleSubmission\HousingKaggle-Navarro.xlsx")


# In[ ]:




