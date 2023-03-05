# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 22:02:37 2021

@author: yeluo
"""

import os
import numpy as np
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn import svm
from sklearn.metrics import roc_curve, auc
import csv
import pandas as pd
import matplotlib.pyplot as plt

###setup working directory
os.chdir("c:/Rwork/risk")

####read data

with open('credit_risk.csv', mode='r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line = 0
    data_temp = []
    for row in csv_reader:
        if line == 0:
            print("variable names",",".join(row))
            var_names = row
            line=line+1
        else:
            data_temp.append(list(map(float,row)))
            line=line+1

csv_file.close()

data_temp_m = np.asmatrix(data_temp)
###data_temp_m is a 13982 by 10 matrix.

df = pd.DataFrame(np.array(data_temp), columns=var_names)

##specify the lr class
LR = LogisticRegression()


###simple example: predictors include income and past_bad_credit
X=df[['income','past_bad_credit']]

y=df['default_label']


###run logistic regression
lr_model = LR.fit(X,y)

###another way to run logistic regression

lr_model1 = sm.Logit(y,sm.add_constant(X)).fit()

###get a summary result of lr
print(lr_model1.summary())


###this is a two dimensional vector, prob d=0 and prob d=1, use the second one
predicted_prob = lr_model.predict_proba(X)
predicted_default_prob= predicted_prob[:,1]
###

###compute false positive rate and true positive rate using roc_curve function
fpr, tpr, _ = roc_curve(y, predicted_default_prob)
roc_auc = auc(fpr, tpr)


###make a plot of roc curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
