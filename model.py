import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib

df = pd.read_csv ('transaction_dataset.csv')

df = df[['Avg min between sent tnx', 'Avg min between received tnx', 'avg val received', 'avg val sent', 'FLAG']]

df = df.dropna()

X = df.drop('FLAG',axis = 1)
y = df.FLAG

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4) #random_state allows for the RNG to be deterministic for reproduction and testing 

dt = DecisionTreeClassifier() #Decision Tree classifer object
dt.fit(X_train,y_train) #Train by fitting to training set
y_pred = dt.predict(X_test) #Predict fraud/label/'FLAG' on the test set

dt_rec = metrics.recall_score(y_test, y_pred)
print("Recall:",dt_rec)

model_filename = 'model.pkl'
joblib.dump(dt, model_filename)