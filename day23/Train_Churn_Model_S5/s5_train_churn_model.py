#!/usr/bin/env python
# coding: utf-8

# In[30]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


# In[31]:


data = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'


# In[32]:


get_ipython().system('wget $data -O data-week-3.csv')


# In[33]:


df = pd.read_csv('data-week-3.csv')
df.head()


# In[34]:


df.head().T


# In[35]:


df.columns


# In[36]:


df.columns = df.columns.str.lower().str.replace(' ','_')


# In[37]:


df.columns


# In[38]:


df.head().T


# In[39]:


categorical_columns = list(df.dtypes[df.dtypes=='object'].index)


# In[40]:


categorical_columns


# In[41]:


df['gender']


# In[42]:


for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ','_')


# In[43]:


df.head().T


# In[44]:


df.dtypes


# In[45]:


df.totalcharges


# In[46]:


df.totalcharges = pd.to_numeric(df.totalcharges,errors='coerce')


# In[47]:


df.totalcharges.isnull()


# In[48]:


df[df.totalcharges.isnull()][['customerid','totalcharges']]


# In[49]:


df.totalcharges = df.totalcharges.fillna(0)


# In[50]:


df[df.totalcharges.isnull()][['customerid','totalcharges']]


# In[51]:


df.churn


# In[52]:


df.churn = (df.churn=='yes').astype(int)


# In[53]:


df.churn


# Setting up Validation framework
# 

# In[54]:


from sklearn.model_selection import train_test_split


# In[55]:


df_full_train,df_test = train_test_split(df,test_size = 0.2,random_state = 1)


# In[56]:


len(df_full_train),len(df_test)


# In[57]:


df_train,df_val = train_test_split(df_full_train,test_size=0.25,random_state=1)


# In[58]:


len(df_train),len(df_val),len(df_test)


# In[59]:


#resetindex
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

#target_variables
y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values

#delete_target_values_from_dataframe
del df_train['churn']
del df_val['churn']
del df_test['churn']



# EDA - NUMERICAL AND CATEGORICAL VARIABLES

# In[60]:


df_full_train = df_full_train.reset_index(drop=True)


# In[61]:


df_full_train.churn.value_counts(normalize=True)


# In[62]:


global_churn_rate = df_full_train.churn.mean()


# In[63]:


round(global_churn_rate,2)


# In[64]:


df_full_train.dtypes


# In[65]:


numerical = ['tenure','monthlycharges','totalcharges']


# In[66]:


categorical =['gender', 'seniorcitizen', 'partner', 'dependents','phoneservice',
        'multiplelines', 'internetservice',
        'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
        'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
        'paymentmethod'] 


# In[67]:


df_full_train[categorical].nunique()


# Feature Importance: Churn Rate And Risk Ratio

# Churn rate

# In[68]:


df_full_train.head()


# In[69]:


churn_female = df_full_train[df_full_train.gender=='female'].churn.mean()
churn_female


# In[70]:


churn_male = df_full_train[df_full_train.gender=='male'].churn.mean()
churn_male


# In[71]:


churn_with_partner = df_full_train[df_full_train.partner =='yes'].churn.mean()
churn_with_partner


# In[72]:


churn_no_partner = df_full_train[df_full_train.partner =='no'].churn.mean()
churn_no_partner


# In[73]:


global_churn_rate


# Risk Ratio
# 
# 
# Risk = Group/Global
# 
# If Risk >1 - More likely to churn
# If Risk<1 - Less likely to churn

# In[74]:


churn_no_partner/global_churn_rate


# In[75]:


from IPython.display import display


# In[76]:


for c in categorical:
    print(c)
    df_group = df_full_train.groupby(c).churn.agg(['mean','count'])
    df_group['diff'] = df_group['mean'] - global_churn_rate
    df_group['risk'] = df_group['mean']/global_churn_rate
    display(df_group)
    print('\n')
    


# Feature Importance: Mutual Information
# 
# Mutual information- concept from information theory, it tells us how much we can learn about one variable if we know the value of another
# * https://en.wikipedia.org/wiki/Mutual_ information

# In[77]:


from sklearn.metrics import mutual_info_score


# In[78]:


#same regardless of the order of arguments
mutual_info_score(df_full_train.churn, df_full_train.contract)


# In[79]:


mutual_info_score(df_full_train.churn, df_full_train.gender)


# In[80]:


def mutual_info_churn_score(series):
    return mutual_info_score(series, df_full_train.churn)


# In[81]:


mutual_info = df_full_train[categorical].apply(mutual_info_churn_score)


# In[82]:


mutual_info.sort_values(ascending= False)


# In[83]:


mutual_info.gender


# 3.7 Feature importance: Correlation
# 
# How about numerical columns?
# 
# Correlation coefficient- https://en.wikipedia.org/wikiPearson_correlation_coefficient

# In[84]:


df_full_train.tenure.max()


# In[85]:


df_full_train[numerical]


# In[86]:


df_full_train[numerical].corrwith(df_full_train.churn)


# In[87]:


df_full_train[df_full_train.tenure >3].churn.mean()


# 3.8 One-hot encoding
# 
# 
# Use Scikit-Learn to encode categorical features

# In[88]:


#one-hot-encoding with sklearn
from sklearn.feature_extraction import DictVectorizer


# In[89]:


df_train[['gender','contract']].iloc[:100]


# In[90]:


#converting to dictionary
train_dicts = df_train[categorical + numerical].to_dict(orient='records')


# In[91]:


train_dicts[0]


# In[92]:


dv = DictVectorizer(sparse = False)


# In[93]:


dv.fit(train_dicts)
X_train = dv.transform(train_dicts)


# In[94]:


dv.get_feature_names()


# In[95]:


X_train


# In[96]:


X_train.shape


# In[97]:


val_dicts = df_val[categorical+numerical].to_dict(orient='records')


# In[98]:


X_val = dv.transform(val_dicts)


# In[99]:


X_val.shape


# 3.9 Logistic regression
# 
# Binary classification
# 
# Linear vs logistic regression

# In[100]:


def sigmoid(z):
    return (1/(1+np.exp(-z)))


# In[101]:


z = np.linspace(-7.5,8,61)


# In[102]:


sigmoid(z)


# In[103]:


#without_sigmoid
def linear_reg(xi):
    result = W0

    for i in range(len(w)):
        result = result + xi[i]*w[i]
    return result


# In[104]:


#with_sigmoid
def linear_reg(xi):
    result = W0

    for i in range(len(w)):
        result = result + xi[i]*w[i]
    score = sigmoid(result)
    return score


# 3.10 Training logistic regression with Scikit-Learn
# 
# Train a model with Scikit-Learn
# 
# Apply it to the validation dataset
# 
# Calculate the accuracy

# In[105]:


from sklearn.linear_model import LogisticRegression


# In[106]:


model = LogisticRegression()


# In[107]:


model.fit(X_train,y_train)


# In[108]:


#bias_term
model.intercept_


# In[109]:


#weights_terms
model.coef_


# In[110]:


y_pred = model.predict_proba(X_val)[:,1]


# In[111]:


y_pred.shape


# In[112]:


churn_decision = y_pred >=0.5


# In[113]:


churn_decision


# In[114]:


churn_decision.shape


# In[115]:


(y_val==churn_decision).mean()


# 4.2 Accuracy and dummy model
# 
# Evaluate the model on different thresholds
# 
# Check the accuracy of dummy baselines

# In[116]:


len(y_val)


# In[117]:


(y_val==churn_decision).sum()


# In[118]:


y_pred


# In[119]:


(y_val==churn_decision).sum()/len(y_val)


# In[120]:


from sklearn.metrics import accuracy_score


# In[121]:


thresholds = np.linspace(0,1,21)

scores = []

for t in thresholds:
    score = accuracy_score(y_val,y_pred>=t).mean()
    print("%.2f %.3f" %(t,score))
    scores.append(score)


# In[122]:


scores


# In[123]:


plt.plot(thresholds,scores)


# In[124]:


from collections import Counter


# In[125]:


Counter(y_pred>=1)


# In[126]:


Counter(y_pred<0.5),Counter(y_pred>=0.5)


# In[127]:


Counter(y_val)


# In[128]:


386/1023


# 4.3 Confusion table
# 
# Different types of erors and correct decisions
# 
# Arranging them in a table

# In[129]:


actual_positive = (y_val ==1)
actual_negative = (y_val ==0)


# In[130]:


sum(actual_positive)


# In[131]:


sum(actual_negative)


# In[132]:


t = 0.5
predict_positve = (y_pred>=t)
predict_negative = (y_pred<t)


# In[133]:


true_positive = (predict_positve & actual_positive).sum()
false_positive =(predict_positve & actual_negative).sum()


# In[134]:


true_negative = (predict_negative & actual_negative).sum()
false_negative = (predict_negative & actual_positive).sum()


# In[135]:


true_positive,false_positive


# In[136]:


true_negative,false_negative


# In[137]:


confusion_matrix = np.array([[true_negative,false_positive],[false_negative,true_positive]]) 


# In[138]:


confusion_matrix


# In[139]:


pd.DataFrame(confusion_matrix,index=['Actual Negative','Actual Positive'],columns = ['Pred Negative','Pred Positive'])


# In[140]:


confusion_matrix/confusion_matrix.sum()


# In[141]:


confusion_matrix


# 4.4 Precision and Recall
# 

# In[142]:


(true_positive+true_negative)/confusion_matrix.sum()


# In[143]:


precision = true_positive/(true_positive + false_positive)


# In[144]:


precision


# In[145]:


recall = true_positive/(true_positive+false_negative)


# In[146]:


recall


# In[147]:


#percentage of population we failed to identify as churning

1 -recall


# ## 4.5 ROC Curves
# 

# TPR and FRP

# In[148]:


tpr = true_positive/(true_positive + false_negative)
tpr


# In[149]:


fpr = false_positive/(false_positive + true_negative)
fpr


# In[150]:


scores = []

thresholds = np.linspace(0,1,101)

for t in thresholds:
    actual_positive =  (y_val==1)
    actual_negative = (y_val==0)

    predict_positve = (y_pred>=t)
    predict_negative = (y_pred<t)

    tp= (predict_positve & actual_positive).sum()
    tn = (predict_negative & actual_negative).sum()

    fp = (predict_positve & actual_negative).sum()
    fn  = (predict_negative & actual_positive).sum ()

    scores.append((t,tp,fp,fn,tn))

scores


# In[151]:


columns = ['threshold','tp','fp','fn','tn']
df_scores = pd.DataFrame(scores,columns = columns)




df_scores


# In[152]:


#print incremented values from the dataframe df_scores
df_scores[::10]


# In[153]:


df_scores['tpr'] = df_scores.tp/(df_scores.tp + df_scores.fn)
df_scores['fpr'] = df_scores.fp/(df_scores.fp + df_scores.tn)


# In[154]:


df_scores


# In[155]:


plt.plot(df_scores.threshold,df_scores['tpr'],label='TPR')
plt.plot(df_scores.threshold,df_scores['fpr'],label='FPR')
plt.legend()
plt.show()


# ## Random Model
# 

# In[156]:


np.random.seed(1)
y_rand = np.random.uniform(0,1,size=len(y_val))
y_rand.round(3)


# In[157]:


((y_rand>=0.5) == y_val).mean()


# In[158]:


def tpr_fpr_df(y_val,y_pred):
    scores = []

    thresholds = np.linspace(0,1,101)

    for t in thresholds:
        actual_positive =  (y_val==1)
        actual_negative = (y_val==0)

        predict_positve = (y_pred>=t)
        predict_negative = (y_pred<t)

        tp= (predict_positve & actual_positive).sum()
        tn = (predict_negative & actual_negative).sum()

        fp = (predict_positve & actual_negative).sum()
        fn  = (predict_negative & actual_positive).sum ()

        scores.append((t,tp,fp,fn,tn))
    
    columns = ['threshold','tp','fp','fn','tn']
    df_scores = pd.DataFrame(scores,columns = columns)

    df_scores['tpr'] = df_scores.tp/(df_scores.tp + df_scores.fn)
    df_scores['fpr'] = df_scores.fp/(df_scores.fp + df_scores.tn)

    return df_scores
        


# In[159]:


df_rand = tpr_fpr_df(y_val,y_rand)


# In[160]:


df_rand[::10]


# In[161]:


plt.plot(df_rand.threshold,df_rand['tpr'],label='TPR')
plt.plot(df_rand.threshold,df_rand['fpr'],label='FPR')
plt.legend()
plt.show()


# Idea model

# In[162]:


num_neg = (y_val ==0).sum()
num_pos = (y_val ==1).sum()
num_neg,num_pos


# In[163]:


y_ideal = np.repeat([0,1],[num_neg,num_pos])
y_ideal


# In[164]:


y_ideal_pred = np.linspace(0,1,len(y_val))
y_ideal_pred


# In[165]:


(y_val==0).mean()


# In[166]:


(y_ideal_pred>=0.726).mean()


# In[167]:


y_ideal.mean()


# In[168]:


((y_ideal_pred>=0.726)==y_ideal).mean()


# In[169]:


df_ideal = tpr_fpr_df(y_ideal,y_ideal_pred)


# In[170]:


df_ideal[::10]


# In[171]:


plt.plot(df_ideal.threshold,df_ideal['tpr'],label='TPR',color = 'black')
plt.plot(df_ideal.threshold,df_ideal['fpr'],label='FPR',color = 'black')


#plt.plot(df_rand.threshold,df_rand['tpr'],label='TPR')
#plt.plot(df_rand.threshold,df_rand['fpr'],label='FPR')


plt.plot(df_scores.threshold,df_scores['tpr'],label='TPR')
plt.plot(df_scores.threshold,df_scores['fpr'],label='FPR')

plt.legend()


# In[172]:


plt.figure(figsize=(5,5))

plt.plot(df_scores.fpr,df_scores.tpr,label='Model')
plt.plot([0,1],[0,1],label='Random',linestyle='--')

plt.xlabel('FPR')
plt.ylabel('TPR')

plt.legend()


# In[173]:


from sklearn.metrics import roc_curve


# In[174]:


fpr,tpr,thresholds= roc_curve(y_val,y_pred)


# In[175]:


plt.figure(figsize=(5,5))

plt.plot(fpr,tpr,label='Model')
plt.plot([0,1],[0,1],label='Random',linestyle='--')

plt.xlabel('FPR')
plt.ylabel('TPR')

plt.legend()


# ## ROC AUC

# Area under the ROC curve - useful metric
# 
# 
# Interpretation of AUC

# In[176]:


from sklearn.metrics import auc


# In[177]:


auc(fpr,tpr)


# In[178]:


auc(df_scores.fpr,df_scores.tpr)


# In[179]:


auc(df_ideal.fpr,df_ideal.tpr)


# In[180]:


fpr,tpr,thresholds= roc_curve(y_val,y_pred)


# In[181]:


auc(fpr,tpr)


# In[182]:


from sklearn.metrics import roc_auc_score


# In[183]:


roc_auc_score(y_val,y_pred)


# In[184]:


import random


# In[185]:


pos = y_pred[(y_val==1)]
neg = y_pred[(y_val==0)]


# In[186]:


pos


# In[187]:




n = 100000
success = 0 

for i in range(n):
    pos_ind = random.randint(0, len(pos) - 1)
    neg_ind = random.randint(0, len(neg) - 1)

    if pos[pos_ind] > neg[neg_ind]:
        success = success + 1

success / n


# In[188]:


pos_ind = random.randint(0,len(pos)-1)
neg_ind = random.randint(0,len(neg)-1)

pos[pos_ind] > neg[neg_ind]


# In[189]:


pos_ind,neg_ind


# In[190]:


n = 50000

np.random.seed(1)
pos_ind = np.random.randint(0, len(pos), size=n)
neg_ind = np.random.randint(0, len(neg), size=n)

(pos[pos_ind] > neg[neg_ind]).mean()


# ## 4.7 Cross-Validation

# Evaluating the same model on different subsets of data
# 
# 
# Getting the average prediction and the spread within predictions.

# In[191]:


def train(df_train,y_train,C):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C,max_iter=1000)
    model.fit(X_train,y_train)

    return dv,model


# In[192]:


dv,model = train(df_train,y_train,C=10)


# In[193]:


def predict(df_val,dv,model):
    dicts = df_val[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)

    y_pred = model.predict_proba(X)[:,1]

    return y_pred


# In[194]:


from sklearn.model_selection import KFold


# In[195]:


from tqdm.auto import tqdm


# In[196]:



C=1.0
n_splits = 5

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.churn.values
    y_val = df_val.churn.values

    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

    print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))


# In[197]:


scores


# In[198]:


dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, dv, model)

y_test = y_test
auc = roc_auc_score(y_test, y_pred)
auc


# Save the model

# In[199]:


import pickle


# In[200]:


output_file = f'model_C={C}.bin' 
output_file


# In[201]:


f_out= open(output_file,'wb')
pickle.dump((dv,model),f_out)
f_out.close()


# In[202]:


with open(output_file,'wb') as f_out:
    pickle.dump((dv,model),f_out)


# Load the Model

# In[203]:


import pickle


# In[204]:


model_file = 'model_C=1.0.bin'


# In[205]:


with open(model_file,'rb') as f_in:
    dv,model = pickle.load(f_in)
    


# In[206]:


dv,model


# In[207]:


customer = {
  'gender': 'female',
  'seniorcitizen': 0,
  'partner': 'yes',
  'dependents': 'no',
  'tenure': 1,
  'phoneservice': 'no',
  'multiplelines': 'no_phone_service',
  'internetservice': 'dsl',
  'onlinesecurity': 'no',
  'onlinebackup': 'yes',
  'deviceprotection': 'no',
  'techsupport': 'no',
  'streamingtv': 'no',
  'streamingmovies': 'no',
  'contract': 'month-to-month',
  'paperlessbilling': 'yes',
  'paymentmethod': 'electronic_check',
  'monthlycharges': 29.85,
  'totalcharges': 29.85,
}


# In[208]:


X = dv.transform([customer])


# In[209]:


model.predict_proba(X)[:,1]

