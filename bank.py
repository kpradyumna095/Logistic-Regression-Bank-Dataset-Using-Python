# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 12:20:40 2019

@author: Hello
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

bank = pd.read_csv("C:\\Users\\Hello\\Desktop\\Data science\\data science\\assignments\\logisticreg\\datasets\\bank-full.csv", sep=";")

bank.job.value_counts()
bank.marital.value_counts()
bank.education.value_counts()
bank.default.value_counts()
bank.housing.value_counts()
bank.loan.value_counts()
bank.contact.value_counts()
bank.month.value_counts()
bank.poutcome.value_counts()
bank.pdays.value_counts()

bank1 = pd.get_dummies(bank, drop_first = True)

plt.boxplot(bank1["age"])
age= np.cbrt(bank1["age"])
plt.boxplot(np.cbrt(bank1["age"]))
plt.boxplot(np.log(bank1["age"]))
ag_log =pd.DataFrame(np.log(bank1["age"]))
ag_log.rename(columns={"age":"age_log"},inplace=True)

bank1 = pd.concat([ag_log,bank1],axis=1)
bank1.drop(["age"],inplace = True, axis=1)
#bank1.drop(["job","marital","education","default","housing","loan","contact","month","poutcome"],inplace = True, axis=1)

#y = pd.get_dummies(bank1["y"],drop_first = True)

#bank1 = pd.concat([bank,bank_dummies], axis=1)

x= bank1.iloc[:,0:42]
#x.drop(["y"],inplace= True , axis=1)

import statsmodels.formula.api as sm

model1= sm.logit("y_yes~x", data = bank1 ).fit()
model1.summary()
model1.summary2()
## AIC: 21648.2702

## 5, 23,19 are majorly insignificant. The p value is more than 0.900
## 23- default_yes is being removed and seeing if the insignificance of the variables are removed 

x.iloc[:,23].name
x0= x.drop(["default_yes"],axis=1)

model2= sm.logit("y_yes~x0", data = bank1).fit()
model2.summary()
model2.summary2()
 ## AIC: 21642
 
 ## 5 from x0 is removed as it is insignificant
#pdays: number of days that passed by after the client was last contacted from a previous campaign 
x0.iloc[:,5].name
x1 = x0.drop(["pdays"],axis=1) 
model3 = sm.logit("y_yes~x1", data =  bank1).fit()
model3.summary()
model3.summary2()
##AIC  21640

## 18 from x1 is removed as it is insignificant
x1.iloc[:,18].name
x2 = x1.drop(["marital_single"],axis=1)
model4 = sm.logit("y_yes~x2", data = bank1).fit()
model4.summary()
model4.summary2()
##AIC  21639

## 38 from x2 is removed as it is insignificant
#poutcome: outcome of the previous marketing campaign 
x2.iloc[:,38].name
x3 = x2.drop(["poutcome_unknown"],axis=1)
model5 = sm.logit("y_yes~x3", data = bank1).fit()
model5.summary()
model5.summary2()
## AIC 21638
### poutcome_unknown removed with the same logic of dummy variables. Out of 4 dummy variables, one can be removed.
## As, the model understands with 3 dummy variables.

## 16 from x3 is removed as it is insignificant
x3.iloc[:,16].name
x4 = x3.drop(["job_unknown"],axis=1)
model6 = sm.logit("y_yes~x4", data= bank1).fit()
model6.summary()
model6.summary2()
##AIC 21638

x4.iloc[:,15].name ## from the model6 summary if we see, x4[15] is job_unemployed
## job umemployed is insignificant in job variable but job is very important for this. In job variable all the categories are significant except unknown and umemployed.
## unknown is removed as the model understands. i.e job has 12 variables and if we create 11 variables, then the model understands the 12th dummy variable.
## As, out of 11, 10 dummy variables are significant and only one is insignificant. I consider the job variable.

x4.iloc[:,26].name
### in the same way in month variables, we create 12 dummy variables. In that one dummy is insignificant.
## Feb is insignificant.
## So, out of 12 months i.e 12 dummy variables, Feb dummy is removed as the model understands.

x5 = x4.drop(["month_feb"],axis=1)
model7 = sm.logit("y_yes~x5", data= bank1).fit()
model7.summary()
model7.summary2()
##AIC  21639

## 5 in x5 is removed, as it is insignificant 
x5.iloc[:,5].name
## previous is the variable. As poutcome,pdays are removed as it was insignificant.
## previous is also interlinked to poutcome, pdays
##previous: number of contacts performed before this campaign and for this client. So, previous is removed
## ( Assumption being explained model5)

x6= x5.drop(["previous"], axis=1)
model8 = sm.logit("y_yes~x6", data = bank1).fit()
model8.summary()
model8.summary2()
## AIC 21641
#### Both pdays and previous.
#pdays: number of days that passed by after the client was last contacted from a previous campaign.
##previous: number of contacts performed before this campaign and for this client.
 

###### Assumption on previous
## My assumption is that contacts performed before this campaign won't be effecting the customer, intaking up the whether the customer will opt for term loan or not.
## As, the contacts made on present campaign should be more stronger. We can use contacts performed in previous campaign.
## in order to improve the stratergy this time for the conversion rate.  

###### Assumption on pdays
## Even number of days that passed by after the client was last contacted will also not effect, because
## we will be more focusing on the present campaign (fresh campaign) and if we look at the data, most of the clients are not previously contacted.
 

###### Assumption on poutcome.
## My assumption is that poutcome i.e outcome of the previous campaign can be considered.Because
## we can to understand, the result of the previous campaign and increase much more success in outcome.


##### Model selection
## All the variables are significant.
## As all the variabls are significant, but the AIC of model8 is not the least.
## but variables are insignificant.

## So, considering the model8, we proceed

y_yes = pd.DataFrame(bank1["y_yes"])

y_yes.rename(columns={"y_yes":"y_yes1"},inplace= True)

x6 = pd.concat([y_yes,x6],axis=1)

from scipy import stats
import scipy.stats as st

st.chisqprob = lambda chisq, df:stats.chi2.sf(chisq,df)

y_pred = model8.predict(x6)
#y_pred1 = model8.predict(bank1)

x6["pred_prob"]= y_pred
x6["y_val"]= np.zeros(45211)
x6.loc[y_pred>=0.5,"y_val"]=1

from sklearn.metrics import classification_report
classification_rep = classification_report(x6["y_val"],x6["y_yes1"])

## confusion matrix
confusion_matrix= pd.crosstab(x6["y_yes1"],x6["y_val"])

##accuracy
accuracy = (38944+1829)/45211 ## 90.2%

## ROC curve
from sklearn import metrics
##fpr=> false postive rate
##tpr=> true positive rate 

fpr,tpr,threshold = metrics.roc_curve(x6["y_yes1"],y_pred)

## plotting roc curve
plt.plot(fpr,tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

## Area under the curve

roc_auc = metrics.auc(fpr,tpr) ##90.76%

### trail 2 
y_pred2= model8.predict(x6)

###
x6["pred_prob2"]=y_pred2
x6["y_val2"]= np.zeros(45211)

x6.loc[y_pred2>=0.82,"y_val2"]=1
## Classification report
classification_rep2 = classification_report(x6["y_val2"],x6["y_yes1"])

##confusion matrix
confusion_matrix2 = pd.crosstab(x6["y_yes1"],x6["y_val2"])

##accuracy
accuracy2= (39651+664)/45211## 90.2%

## The cutoff point is 0.5, above which the accuracy decreases,

x6.drop(["pred_prob","y_val","pred_prob2","y_val2"],inplace= True, axis=1)
x6 = x6.iloc[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,0]]
#x6.rename(columns={"job_self-employed":"job_self","job_blue-collar":"job_blue"},inplace=True)
## Splitting the data into train and test data

x6.rename(columns={"y_yes1":"y_yes"},inplace=True)
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(x6)
## renaming y_yes1 as y_yes from data set x6. So, changing in train_data, test_data 
#train_data.rename(columns={"y_yes1":"y_yes"},inplace=True)
#test_data.rename(columns={"y_yes1":"y_yes"},inplace = True)

### As we are getting an error of mismatch of rows between X6 and train data, we modify the model as,

x_train = train_data.iloc[:,1:36]
##training data
model81 = sm.logit("y_yes~x_train", data = train_data).fit()
model81.summary()
## prediction on training data
y_train = model81.predict(train_data)

train_data["pred_train"]=y_train
train_data["train_vals"]=np.zeros(33908)
train_data.loc[y_train>=0.5,"train_vals"]=1

##confusion matrix
confusion_matrix_train = pd.crosstab(train_data["y_yes"],train_data["train_vals"])

## accuracy

accuracy = (29166+1419)/33908
### 90.2%

##prediction on test data
x_test = test_data.iloc[:,1:36]

model82 = sm.logit("y_yes~x_test", data = test_data).fit()
model82.summary()

## prediction
y_test = model82.predict(test_data)
test_data["pred_test"]=y_test
test_data["test_vals"] = np.zeros(11303)
test_data.loc[y_test>=0.5,"test_vals"]=1

## confusion matrix

confusion_matrix_test = pd.crosstab(test_data["y_yes"],test_data["test_vals"])

## Accuracy
accuracy_test= (9783+419)/11303
##90.3%