import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
titanic_data = pd.read_csv('train_and_test2.csv')
print(titanic_data.head())
print(titanic_data.isnull().sum())
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0],inplace=True)
print(titanic_data.isnull().sum())
x=titanic_data.drop(columns=['Passengerid','Survived'],axis=1)
y=titanic_data['Survived']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
model=LogisticRegression()
model.fit(x_train,y_train)
x_train_prediction=model.predict(x_train)
training_data_accuracy=accuracy_score(y_train,x_train_prediction)
print('Accuracy score of training data : ', training_data_accuracy)
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(y_test, x_test_prediction)
print('Accuracy score of test data : ', test_data_accuracy)
input_data=(35,8.09,1,0,0,3,2.0)
input_data=np.array(input_data)
input_data_reshaped = input_data.reshape(1,-1)
print(input_data_reshaped)
prediction = model.predict(input_data_reshaped)
#print(prediction)
if prediction[0]==0:
    print("Dead")
if prediction[0]==1:
    print("Alive")



