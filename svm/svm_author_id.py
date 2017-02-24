#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
from tools.email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

#features_train = features_train[:len(features_train) // 100]
#labels_train = labels_train[:len(labels_train) // 100]



#########################################################
### your code goes here ###
from sklearn.svm import SVC
clf = SVC(C=10000,kernel='rbf')
trainBeforeTime = time()
clf.fit(features_train,labels_train)
print("training time:", round(time()- trainBeforeTime, 3), "s")

predictBeforeTime = time()
pred=clf.predict(features_test)
print("predict time:", round(time()- predictBeforeTime, 3), "s")

from sklearn.metrics import accuracy_score

print(accuracy_score(labels_test,pred))

print("第10、26、50預測結果:",pred[10],',',pred[26],',',pred[50])

count=0
for predResult in pred:
    if predResult == 1:
        count+=1
print("預測結果為“Chris” (1) 筆數:",count)

#########################################################


