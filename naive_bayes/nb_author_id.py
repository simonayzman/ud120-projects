#!/usr/bin/python

"""
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project.

    Use a Naive Bayes Classifier to identify emails by their authors

    authors and labels:
    Sara has label 0
    Chris has label 1
"""

import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# Initialize classifier
clf = GaussianNB()

# Train model (and time it)
trainingBeganTime = time()
clf.fit(features_train, labels_train)
trainingFinishedTime = time()

# Predict labels on test data (and determine accuracy)
prediction = clf.predict(features_test)
accuracy = accuracy_score(labels_test, prediction)

# Provide useful metrics
print "Training Time:", round(trainingFinishedTime - trainingBeganTime, 3), "s"
print "Accuracy:", accuracy
