
# coding: utf-8

# In[242]:

# importing Python libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
from sklearn import preprocessing


# In[243]:

d=pd.read_csv(r'C:\Users\sahithi\Desktop\Mith_Insofe\dataset.csv')
data=d.copy()


# In[244]:

#Data exploration
data.shape


# In[245]:

data.columns


# In[230]:

data.dtypes


# In[246]:

data['ExtraTime'].replace('Yes','1',inplace=True)
data['ExtraTime'].replace('No','0',inplace=True)


# In[247]:

#Target distribution 
target=data['ExtraTime']
target.value_counts() 


# In[234]:

#Data Preparation
#dropping columns which have unique levels, and less variance and high cardinality
data.drop(['RowID','EmployeeID','EmployeeCount','Over18','datacollected','StandardHours','Joblevel','FirstJobDate','DateOfjoiningintheCurrentCompany','ExtraTime'],axis=1,inplace=True)
data.shape


# In[250]:

#Numeric and categorical
#basic cleaning 
df=data.dtypes
cat_cols=list(df[df=='object'].index)
num_cols=list(df[df.index.difference(cat_cols)].index)
print("categorical columns",cat_cols)
print("numerical columns",num_cols)


# In[235]:

data=pd.get_dummies(data)
data.shape


# In[240]:

#Data Preparation
df_train = data[data['istrain']==1]
df_test = data[data['istrain']==0]


# ##There is no class imbalance

# In[ ]:

tr=df_train[num_cols]
ts=df_test[num_cols]
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
train_scaled=pd.DataFrame(scaler.fit_transform(tr),columns=num_cols)
test_scaled=pd.DataFrame(scaler.fit_transform(ts),columns=num_cols)


# In[239]:

cat_data=df_train[cat_cols]
cat_data2=df_test[cat_cols]
cat_data.shape


# In[141]:

train_data=pd.concat([train_scaled,cat_data1],axis=1)
test_data=pd.concat([test_scaled,cat_data2],axis=1)


# In[142]:

data.head(5)


# In[143]:

data=pd.get_dummies(data)
data.shape


# In[144]:

data.head(5)


# In[145]:

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV # Search over specified parameter values for an estimator.
from sklearn.model_selection import RandomizedSearchCV # Search over specified parameter values for an estimator.
from sklearn.model_selection import ShuffleSplit # Random permutation cross-validator
from sklearn.metrics import make_scorer,recall_score,accuracy_score
from time import time


# In[ ]:

X_train, X_test, y_train, y_test =train_test_split(df_train,target,test_size=0.3)


# In[25]:

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[27]:

#Logistic regression
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,y_train)


# In[28]:

l_predict=model.predict(X_test)
accuracy_score(y_test,l_predict)


# In[ ]:

from sklearn.linear_model import LogisticRegression
model=LogisticRegression(penalty='l2',C=0.01)
model.fit(X_train,y_train)


# In[31]:

l_predict=model.predict(X_test)
accuracy_score(y_test,l_predict)


# In[19]:

#Model Building
from sklearn.svm import SVC


# In[ ]:

#SVM Classifier
scoring={'Accuracy':accuracy_score, 'Recall':recall_score}
Cs=np.arange(0.001,10,0.001)
gamma=np.arange(0.01,1,0.01)
ks=['linear']
param_grid={'C':Cs,'gamma':gamma, 'kernel':ks}
clf_linear=RandomizedSearchCV(SVC(class_weight='balanced'),param_grid,n_jobs=-1,n_iter=10,refit="Recall",verbose=1)
clf_linear.fit(X_train,y_train)
svm_opt=clf.linear.best_estimator_


# In[174]:

#RandomForest Classifer
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(max_depth=16, min_samples_leaf= 20, n_estimators= 100)
clf.fit(X_train,y_train)


# In[175]:

clf_predict=clf.predict(X_test)
print('accuracy',accuracy_score(y_test,clf_predict))


# In[176]:

y_pred_final = clf.predict(df_test)


# In[ ]:

#optimising hyperparameters
start=time()
rf_classifier=RandomForestClassifier(random_state=42)
cv_set=ShuffleSplit(random_state=4)
parameters={'n_estimators':[100,120,140],'min_samples_leaf':[10,20,30],'max_depth':[10,15,20]}
scorer={'Accuracy':accuracy_score, 'Recall':recall_score}
scorer=make_scorer(accuracy_score)
grid_obj=RandomizedSearchCV(rf_classifier,parameters,n_iter=10,cv=cv_set,scoring=scorer)
grid_obj.fit(X_train,y_train)
rf_opt=grid_obj.best_estimator_
end=time()
time=(end-start)/60
print('Rf time',time)


# In[143]:

grid_obj.best_params_


# In[148]:

rf_predict=rf_opt.predict(X_test)
print('accuracy',accuracy_score(y_test,rf_predict))


# In[149]:

importances = rf_opt.feature_importances_
X_train.columns.values[(np.argsort(importances)[::-1])[:5]]


# In[150]:

rf_opt_preds = rf_opt.predict(X_test)


# In[151]:

y_pred_final = rf_opt.predict(df_test)


# In[152]:

data1=pd.read_csv(r'C:\Users\sahithi\Desktop\Mith_Insofe\dataset.csv')
data1.drop(['EmployeeID','EmployeeCount','Over18','datacollected','StandardHours','Joblevel','FirstJobDate','DateOfjoiningintheCurrentCompany','ExtraTime'],axis=1,inplace=True)
test_data = data1[data1['istrain']==0]
test_data.shape


# In[153]:

test_data.columns


# In[177]:

# Final submission
my_submission = pd.DataFrame({'RowId': test_data.RowID, 'ExtraTime': y_pred_final})
my_submission.to_csv('PRED3.csv', index=False)

