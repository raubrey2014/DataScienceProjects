#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
file      gender_classifier.py
author    Ryan Aubrey <rma7qb@virginia.edu>
version   1.0
date      May 29, 2017


brief     Example classification of gender based on height, weight, shoe size using several classifiers
usage     python gender_classifier.py
"""

from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC #Support Vector Classifier
import numpy as np

clf = tree.DecisionTreeClassifier()
neighbor_clf = KNeighborsClassifier(n_neighbors=3)
random_forest_clf = RandomForestClassifier(n_estimators=3)
svc_clf = SVC(probability=True)


# [height, weight, shoe_size]
X = np.array([[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]])

Y = np.array(['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male'])

clf = clf.fit(X, Y)
neighbor_clf = clf.fit(X, Y)
random_forest_clf = random_forest_clf.fit(X, Y)
svc_clf = svc_clf.fit(X, Y)

data_to_predict = np.array([170, 60, 45]).reshape(-1, 3)

prediction = clf.predict(data_to_predict)
neighbor_prediction = neighbor_clf.predict(data_to_predict)
random_forest_prediction = random_forest_clf.predict(data_to_predict)
svc_prediction = svc_clf.predict(np.array(data_to_predict))

print(prediction)
probability_of_prediction = clf.predict_proba(np.array([170, 60, 45]).reshape(-1,3))
print("Probability of Classification Tree:", probability_of_prediction)

print(neighbor_prediction)
probability_of_prediction = neighbor_clf.predict_proba(np.array([170, 60, 45]).reshape(-1,3))
print("Probability of KNN:", probability_of_prediction)

print(random_forest_prediction)
probability_of_prediction = random_forest_clf.predict_proba(np.array([170, 60, 45]).reshape(-1,3))
print("Probability of Random Forest:", probability_of_prediction)

print(svc_prediction)
probability_of_prediction = svc_clf.predict_proba(np.array([170, 60, 45]).reshape(-1,3))
print("Probability of SVC:", probability_of_prediction)

