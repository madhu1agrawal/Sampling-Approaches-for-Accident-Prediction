#!/usr/bin/env python3

from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import metrics
import pandas as pd
import time

# starting time
start = time.time()
print(time.asctime(time.localtime(start)))

train_csv = '../dataset_creation/N10/train_N10.csv'
test_csv = '../dataset_creation/N10/test_N10.csv'

train = pd.read_csv(train_csv)
test = pd.read_csv(test_csv)

x_train = train.drop(columns = ['osm_id', 'class'])
y_train = train['class']

print(x_train.shape)
print(y_train.shape)

x_test = test.drop(columns = ['osm_id', 'class'])
y_test = test['class']

print(x_test.shape)
print(y_test.shape)

clf = BalancedRandomForestClassifier()

print("Training started")
clf.fit(x_train, y_train)

print("Making Predictions")
y_pred = clf.predict(x_test)

print("AUC Score:")
print(roc_auc_score(y_test, y_pred))

print("Accuracy Score")
print(accuracy_score(y_test, y_pred))

print("Precision Score")
print(precision_score(y_test, y_pred))

print("Recall Score")
print(recall_score(y_test, y_pred))

print("f1 Score")
print(f1_score(y_test, y_pred))

# end time
end = time.time()
print(time.asctime(time.localtime(end)))

# total time taken
print(f"Runtime of the program is {end - start}")
