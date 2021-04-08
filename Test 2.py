import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


df = pd.read_excel (r'C:\Users\Owen\Desktop\UNIVERSITY\Year 3\SEM2\KV6003 Induvidual Projects\DatasetPy.xlsx', sheet_name='EPOCH1', header=1, index_col='Year', usecols="A,B,C,E,F,H,I", skipfooter=4)
df2 = pd.read_excel (r'C:\Users\Owen\Desktop\UNIVERSITY\Year 3\SEM2\KV6003 Induvidual Projects\DatasetPy.xlsx', sheet_name='EPOCH2', header=1, index_col='Year', usecols="A,B,C,E,F,H,I", skipfooter=3)
df3 = pd.read_excel (r'C:\Users\Owen\Desktop\UNIVERSITY\Year 3\SEM2\KV6003 Induvidual Projects\DatasetPy.xlsx', sheet_name='EPOCH3', header=1, index_col='Year', usecols="A,B,C,E,F,H,I", skipfooter=0)
#uncomment for analysis of other iterations
print(df)
#print(df2)
#print(df3)

#Logistic Regression Model
#predicting Red Pixel data based off existing data 

#Data  
X = df[["Green pixel", "Blue pixel"]]
print(X)

#labels
y = df[["Red Pixel"]]
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.5)
X_train.shape
print(X_train)

clf = LogisticRegression(random_state=0).fit(X_train,y_train)
prediction = clf.predict(X_test)
print(y_test.values[0])
print(prediction)

for i in range(len(prediction)):
    print(f"Predicted value: {prediction[i]}, Actual value: {y_test.values[i][0]}, Accuracy={y_test.values[i][0]/prediction[i]}")


#Naive Bayes classifier
#Predicting Red Pixel Data based off year. Will predict the value of deforestion (Red Pixel) in years passed 2019. 

#Data
X = df[["Red Pixel"]]
print(X)

#Labels
y = df.index
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.5)
X_train.shape
print(X_train)

clf = GaussianNB(random_state=0).fit(X_train,y_train)
predictions = clf.predict(X_test)
print(y_test.values[0])
print(predictions)

for i in range(len(predictions)):
    print(f"Predicted value: {predictions[i]}, Actual value: {y_test.values[i][0]}, Accuracy={y_test.values[i][0]/predictions[i]}")
