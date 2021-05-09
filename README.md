# data-science-projects

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Linear regression
data = pd.read_csv("C:\\ML\\Advertising.csv")
data.isnull().sum()

#Input variable
X=data.iloc[:,1:2].values
#Output variable
y=data.iloc[:,4].values

#Spilt data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=1)

#Model Building & Prediction
from sklearn.linear_model import LinearRegression
reg= LinearRegression()
reg.fit(X_train,y_train)
y_pred= reg.predict(X_test)

#Model Evaluation
print(" The accuracy of model on training dataset is {:.4f} ".format(reg.score(X_train,y_train)))
print(" The accuracy of model on testing dataset is {:.4f} ".format(reg.score(X_test,y_test)))

#Coefficient and intercept values
print(reg.coef_)
print(reg.intercept_)

x=100
y=reg.intercept_+(reg.coef_*x)

#Cross validation
from sklearn.model_selection import cross_val_score
scores=cross_val_score(reg,X_train,y_train,scoring='r2',cv=10)
mean_r2= np.mean(scores)
round(mean_r2,2)

#Scatter plot
plt.scatter(X_train,y_train,color='green')
plt.plot(X_train,reg.predict(X_train),color='blue')
plt.title("TV Advertising")
plt.xlabel('TV')
plt.ylabel('Sales')

plt.scatter(X_test,y_test,color='green')
plt.plot(X_test,reg.predict(X_test),color='blue')
plt.title("TV Advertising")
plt.xlabel('TV')
plt.ylabel('Sales')


#Multivariable Linear Regression
#Input variable
X=data.iloc[:,1:4].values
#Output variable
y=data.iloc[:,4].values

#Spilt data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=1)

#Model Building & Prediction
from sklearn.linear_model import LinearRegression
reg= LinearRegression()
reg.fit(X_train,y_train)
y_pred= reg.predict(X_test)

#Model Evaluation
print(" The accuracy of model on training dataset is {:.4f} ".format(reg.score(X_train,y_train)))
print(" The accuracy of model on testing dataset is {:.4f} ".format(reg.score(X_test,y_test)))

#Coefficient and intercept values
print(reg.coef_.round(3))
print(round(reg.intercept_,3))

#Cross validation
from sklearn.model_selection import cross_val_score
scores=cross_val_score(reg,X_train,y_train,scoring='r2',cv=10)
mean_r2= np.mean(scores)
round(mean_r2,2)


#Adjusted r2
#r2Adj= 1-(1-r2)*((n-1)/(n-p-1)
#where n is no of observations and p is number of predictors

# adjusted r2 for training
X_train.shape
p=X_train.shape[1]
#y=b0+b1x1+b2x2+b3x3

n_train=X_train.shape[0]
r2_Train= reg.score(X_train,y_train)
r2_Train_Adj= 1-(1-r2_Train)*((n_train-1)/(n_train-p-1))
r2_Train_Adj

#Adjusted r2 for testing
n_test=X_test.shape[0]
r2_Test= reg.score(X_test,y_test)
r2_Test_Adj= 1-(1-r2_Test)*((n_test-1)/(n_test-p-1))
r2_Test_Adj

import statsmodels.api as sm
x=sm.add_constant(X_train)
result=sm.OLS(y_train,x).fit()
#Get coeff,r2, adj r2,pvalue,f-stat
result.summary()
#on addition of varibales r2 value will increase irrelevant of importance of variables
# adj r2 will always be less than r2

#here p value for newspsper is greater than 0.05 there this feature can be excluded from the model



#import statsmodels.api as sm
#mod=sm.OLS(y_train,X_train)
#fitMod=mod.fit()
#fitMod.summary2()
#fitMod.summary()
#p_values=fitMod.summary2().tables[1]['P>|t|']


