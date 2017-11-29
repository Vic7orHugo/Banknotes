# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 17:26:53 2017

@author: Necron
"""

# Banknotes Authentication

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('banknotes.csv')
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

for model in range(1,8):

    """ --------------------------- Classification Model Decision Making --------------------------- """
    if (model == 1):
        # Fitting Logistic Regression to the Training Set
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(random_state = 0)
        classifier.fit(X_train, y_train)
    
    if (model == 2):
        # Fitting K-NN to the Training Set
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
        classifier.fit(X_train, y_train)
    
    if (model == 3):    
        # Fitting SVM to the Training set
        from sklearn.svm import SVC
        classifier = SVC(kernel = 'linear', random_state = 0)
        classifier.fit(X_train, y_train)
        
    if (model == 4):
        # Fitting Kernel SVM to the Training set
        from sklearn.svm import SVC
        classifier = SVC(kernel = 'rbf', random_state = 0)
        classifier.fit(X_train, y_train)
        
    if (model == 5):
        # Fitting Naive Bayes to the Training set
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB()
        classifier.fit(X_train, y_train)
    
    if (model == 6):    
        # Fitting Decision Tree to the Training set
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
        classifier.fit(X_train, y_train)
        
    if (model == 7):
        # Fitting Random Forest to the Training set
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
        classifier.fit(X_train, y_train)
    
    """##############################################################################################"""
    
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Performance Evaluation
    tp = cm[1,1] # True Positives
    tn = cm[0,0] # True Negatives
    fp = cm[0,1] # False Positives
    fn = cm[1,0] # False Negatives
    accuracy = (tp + tn) / len(y_test)
    precision = tp / (tp + fp)                                # Measuring exactness
    recall = tp / (tp + fn)                                   # Measuring completeness
    f1_score = 2 * precision * recall / (precision + recall)  # Compromise between Precision and Recall
    
    # Evaluation Matrix
    # model = 6
    if (model == 1):
        models, coefficients = 7, 4;
        em = [[0 for x in range(coefficients)] for y in range(models)] 
    em[model-1] = [accuracy, precision, recall, f1_score]

# Creating the dataframe
collist = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
rowlist = ['Logistic', 'K-NN (5)', 'SVM', 'Kernel SVM (Gaussian)', 'Naive Bayes', 'Decision Tree', 'Random Forest (10)']
EM = pd.DataFrame(em, index = rowlist, columns = collist)

# Writing the Evaluation Matrix on a .csv file
EM.to_csv('EM_python.csv', index=True, header=True, sep=',')   

