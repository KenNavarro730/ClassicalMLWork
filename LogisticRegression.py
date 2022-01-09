#!/usr/bin/env python
# coding: utf-8

# In[4]:


from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_olivetti_faces
fofdata = fetch_olivetti_faces()


# In[6]:


fofdata.keys()


# In[21]:


df = pd.DataFrame(fofdata['data'])
df['labels'] = fofdata['target']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(fofdata['data'], fofdata['target'], test_size = 0.2)
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
vals = logreg.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, vals)


# In[31]:


#With basic Logistic Regression, we were able to get a 96.25% accuracy
#Lets now check recall and precision
from sklearn.metrics import recall_score
recall_score(y_test, vals, average = 'macro', zero_division = 1)


# In[33]:


from sklearn.metrics import precision_score
precision_score(y_test, vals, average = 'macro')


# In[35]:


from sklearn.model_selection import RandomizedSearchCV
params = {
    'penalty':['none','l2','l1','elasticnet'],
    'dual':[True, False],
    'tol':[1e-4, 1e-5,1e-6,1e-2],
    'C':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2],
    'max_iter':[100,150,200,50]
}
from sklearn.linear_model import LogisticRegression
logsolver = LogisticRegression(n_jobs=-1)
model = RandomizedSearchCV(logsolver, params, cv = 5, n_iter = 12)
model.fit(fofdata['data'], fofdata['target'])


# In[37]:


model.cv_results_
model.best_estimator_


# In[40]:


# dfresult = pd.DataFrame(model.cv_results_)
# dfresult
model.best_score_ #This best score drew the original model from 96.25% to 96.5%


# In[41]:


#Now that we have these parameters, lets see if we can improve the score even more by gridsearching over the different solvers
intermediatesolver = LogisticRegression(C=1.1, n_jobs = -1, tol = 1e-05)
parameters = {
    'solver':['newton-cg', 'sag','saga','lbgfs']
}
from sklearn.model_selection import GridSearchCV
testing = GridSearchCV(intermediatesolver, parameters, cv = 5)
testing.fit(fofdata['data'], fofdata['target'])
testdf = pd.DataFrame(testing.cv_results_)


# In[42]:


testdf


# In[43]:


#We were able to get the best overall performance with newtoncg added onto our original best model for a 97.25% test score
#So overall we were able to improve performance by a whopping 1.25%. 
#We found best parameters were newton cg as solver, n_jobs = -1, C= 1.1 tol = 1e-05 with 20% test split


# In[ ]:




