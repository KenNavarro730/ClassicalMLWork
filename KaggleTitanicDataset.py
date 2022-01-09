#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
dftrain = pd.read_csv("train.csv")
dftarget = pd.read_csv("test.csv")


# In[177]:


targettrain = dftrain['Survived']
sex = dftrain[['Sex']]
datatrain['Embarked'] = datatrain['Embarked'].fillna("S")
embarkedoe = oeembarked.fit_transform(datatrain[['Embarked']])
embarkedoe = embarkedoe.reshape(-1,1)
embarkedoe = embarkedoe[:,0].tolist()#One hot encoder made the S = 2, Q = 1, C=0
datatrain = dftrain.drop(['Embarked','Sex','Name','Cabin', 'Survived','Ticket'], axis = 'columns') #Dont need name for obvious reasons, might not need fare either
#we can check accuracy difference after testing it with and without fare, #We also dont need ticket, We are one hot encoding
#We will also drop Cabin
#the sex column
from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder()
oeembarked = OrdinalEncoder()
sexoe = oe.fit_transform(sex)
sexoe = sexoe.reshape(-1,1)
sexoe = sexoe[:,0].tolist()
datatrain['Sex'] = sexoe
#With ordinal encoder, male equals 1 and female equals 0
#We can ordinal encode embarked as well, there were 3 different spots people embarked from
datatrain['Embarked'] = embarkedoe
#We can say early equals before 300, Regular equals between 300 and 600 and late equals after 600 and before 900
#The above refers to PassengerId
allid = np.c_[np.zeros(shape=(1,300)), np.ones(shape=(1,300)), np.full((1,291),2)]
allid = allid.reshape(-1,1)
allid = allid[:,0].tolist()
datatrain['PassengerId'] = allid
datatrain
#Last thing we have to worry about now is the age, we checked info and saw that there exists some NaN values, we can take care
#of this with SimpleImputer
from sklearn.impute import SimpleImputer
method = SimpleImputer(missing_values = np.nan, strategy = 'mean')
newage = method.fit_transform(datatrain[['Age']])
newage = newage.astype(int)
datatrain = datatrain.drop(['Age'], axis = 'columns')
datatrain['Age'] = newage


# In[185]:


# Now we have our seperate train and test data, lets do train test split
from sklearn.model_selection import train_test_split
import numpy as np
X_train, X_test, Y_train, Y_test = train_test_split(datatrain, targettrain, test_size = 0.2)
from sklearn.linear_model import LogisticRegression
params = {
    'penalty':['l2','l1'],
    'tol':[1e-5, 1e-4, 1e-3],
    'C':[0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2],
    'max_iter':[100, 150, 200, 220]
}
from sklearn.model_selection import GridSearchCV
logclf = GridSearchCV(LogisticRegression(n_jobs = -1), params, cv = 5) #Test the hyperparameters and their combos set above
#Make sure to do cross validation at each instance
#Dont forget that with gridsearch cv due to the cross validation we dont need to split the data and test set into 80/20 combos
#So the train test split we did earlier isnt very important.
logclf.fit(datatrain, targettrain)
logclf.best_estimator_
datatrain.info()


# In[186]:


cleanread = pd.DataFrame(logclf.cv_results_)
cleanread


# In[187]:


logclf.best_score_


# In[188]:


logclf.best_estimator_


# In[193]:


#We got 79% accuracy with C=0.2, max_iter = 150, and n_jobs = -1, tol = 1e-05
#Now we are going to try different solvers with this.
diffsolvers = {
    'solver':['saga', 'sag', 'newton-cg','lbfgs'],
    'penalty':['l2', 'none']
}
logregtf = LogisticRegression(C=0.2, max_iter = 150, n_jobs = -1, tol=1e-05)
testingclf = GridSearchCV(logregtf, diffsolvers, cv = 5)
testingclf.fit(datatrain, targettrain)
ftdf = pd.DataFrame(testingclf.cv_results_)


# In[194]:


testingclf.best_estimator_


# In[195]:


testingclf.best_score_


# In[200]:


#lets try different penalties and C values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(datatrain, targettrain, test_size = 0.2)
Logregattempt = LogisticRegression(C=1, max_iter = 150, n_jobs = -1, solver = 'liblinear', tol = 1e-05)
# from sklearn.preprocessing import StandardScaler
# Xtrainscaled = StandardScaler().fit_transform(X_train)
# Xtestscaled = StandardScaler().fit_transform(X_test)
# I want to normalize everything that is a number rather than a classification
X_train


# In[202]:


reglogreg = LogisticRegression()
reglogreg.fit(X_train, y_train)
Predicted = reglogreg.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, Predicted)
#From the original model we were able to increase accuracy of log reg by about 6.2%, rather than by using default logistic 
#reg hyperparameters.


# In[203]:


#Now lets try other classifiers, next time we do grid search, before doing so we should get a baseline accuracy by using the 
#basic model without tuning any hyperparameters as the score we are trying to beat
#The other really nice part is that since we already cleaned up the data we dont have to focus on that aspect again and we can 
#just directly plug in hyperparameters into gridsearch to find most optimal hyperparameter setup
from sklearn.ensemble import RandomForestClassifier
rfcreg = RandomForestClassifier()
rfcreg.fit(X_train, y_train)
rfcregpredict = rfcreg.predict(X_test)
accuracy_score(y_test, rfcregpredict)
#Ok so baseline for random forest classifier is 78.77%, Lets see how we can improve this.


# In[ ]:




