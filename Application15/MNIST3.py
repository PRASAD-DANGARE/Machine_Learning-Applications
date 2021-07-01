'''
Classifier          :   Bagging Using Decision Tree
DataSet             :   mnist.csv
Features            :   Pixel 0 To Pixel 783
Labels              :   0, 1, 2, 3 etc

Training Dataset    :   70% of 42001 Entries
Testing Dataset     :   30% of 42001 Entries

Author              :   Prasad Dangare
Date                :   01 July 2021

Function Name       :   MNIST_Using_DT

'''

#====================
#
# IMPORTS
#
#====================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn. ensemble import RandomForestClassifier, BaggingClassifier

#=====================
#
# Accuracy Operation
#
#=====================

def Bagging_Using_DT():


    data = pd.read_csv('mnist.csv')

    DF_x = data.iloc[:,1:]  # Labels
    DF_y = data.iloc[:,0]   # Pixels

    x_train, x_test, y_train, y_test = train_test_split(DF_x, DF_y, test_size = 0.3, random_state = 5)

    #=====================
    #
    # Using Decision Tree
    #
    #=====================
    
    DT = DecisionTreeClassifier()
    DT.fit(x_train, y_train)

    print("\n----------------------------------------------------\n")

    print("Training Accuracy Using Decision Tree : ", DT.score(x_train, y_train) *100)
    print("Testing Accuracy Using Decision Tree : ", DT.score(x_test,y_test) *100)

    print("\n----------------------------------------------------\n")

    #==================================================
    #
    # Using Random Forest - Ensemble Of Decision Trees
    #
    #==================================================

    RF = RandomForestClassifier(n_estimators = 20)
    RF.fit(x_train, y_train)

    print("Training Accuracy Using Random Forest : ", RF.score(x_train, y_train) *100)
    print("Testing Accuracy Using Random Forest : ", RF.score(x_test ,y_test) *100)

    print("\n----------------------------------------------------\n")

    #===============================
    #
    # Bagging Using Decision Tree
    #
    #===============================

    BG = BaggingClassifier(DecisionTreeClassifier(), max_samples = 0.7, max_features = 1.0, n_estimators = 25)
    BG.fit(x_train, y_train)

    print("Training Accuracy Using Bagging Classifier : ", BG.score(x_train, y_train) *100)
    print("Testing Accuracy Using Bagging Classifier : ", BG.score(x_test, y_test) *100)

    print("\n----------------------------------------------------\n")

#=====================
#
# ENTRY POINT 
#
#=====================

def main():

    print("\n\t----- MNIST Case Study Using Bagging -----\n\t")

    Bagging_Using_DT()

#=====================
#
# CODE STARTER
#
#=====================

if __name__ == "__main__":
    main()
