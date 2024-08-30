# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
step 1.start

step 2.Import the standard Libraries.

step 3.Set variables for assigning dataset values.

step 4.Import linear regression from sklearn.

step 5.Assign the points for representing in the graph.

step 6.Predict the regression for marks by using the representation of the graph.

step 7.Compare the graphs and hence we obtained the linear regression for the given datas.

step 8.stop

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Divyadharshini.A
RegisterNumber: 212222240027 
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)

#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)


```

## Output:

## DataSet:

![image](https://github.com/user-attachments/assets/f77424e5-2b43-4033-b94d-b38aa435271f)



## Head Values:

![image](https://github.com/user-attachments/assets/bcd64b85-9fd3-4bff-ac0a-51ee79708591)

## Tail Values:

![image](https://github.com/user-attachments/assets/82f33716-58cc-49fa-87ce-5b01de80ce5e)

## X and Y Values:

![image](https://github.com/user-attachments/assets/fb92dae9-2334-42a8-b1ba-6086fe0f553d)



## Prediction of X and Y:

![image](https://github.com/user-attachments/assets/c7a11aad-7e0f-4e83-a2c9-503df44bcd73)


## MSE, MAE and RMSE:

![image](https://github.com/user-attachments/assets/f0369ba9-b78e-44bf-9d87-4e7fb7894dde)

## Training Sets:

![image](https://github.com/user-attachments/assets/9be7bd15-f102-4096-8a7e-237783ee3ef9)

## Training Sets:

![image](https://github.com/user-attachments/assets/418b00f4-dec5-4109-bb07-35a0b5e15b3d)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
