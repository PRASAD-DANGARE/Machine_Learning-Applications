'''
Classifier          :   Boosting Using Random Forest Classifier
DataSet             :   mnist.csv
Features            :   Pixel 0 To Pixel 783
Labels              :   0, 1, 2, 3 etc

Training Dataset    :   70% of 42001 Entries
Testing Dataset     :   30% of 42001 Entries

Author              :   Prasad Dangare
Date                :   12 July 2021

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
from sklearn. ensemble import RandomForestClassifier
from sklearn. ensemble import AdaBoostClassifier

#=====================
#
# Accuracy Operation
#
#=====================

def Boosting_Using_DT():

    data = pd.read_csv('mnist.csv')

    df_x = data.iloc[:, 1:]  # Labels
    df_y = data.iloc[:, 0]   # Pixels

    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size = 0.3, random_state = 5)

    #DT = DecisionTreeClassifier(max_depth = 4, random_state = 2)
    DT = DecisionTreeClassifier()
    DT.fit(x_train, y_train)

    print("\n----------------------------------------------------\n")

    print("Training Accuracy Using Decision Classifier : ", DT.score(x_train, y_train) * 100)
    print("Testing Accuracy Using Decision Classifier : ", DT.score(x_test, y_test) * 100)

    print("\n----------------------------------------------------\n")

    #ADT = AdaBoostClassifier(DT, n_estimators = 50,learning_rate = 1)
    #ADT = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators = 50, learning_rate = 1.0, random_state = 3)
    ADT = AdaBoostClassifier(RandomForestClassifier(), n_estimators = 20, learning_rate = 1.0, random_state = 3)

    ADT.fit(x_train, y_train)

    print("Training Accuracy Using Boosting In Decision Classifier : ", ADT.score(x_train, y_train) * 100)
    print("Testing Accuracy Using Boosting In Decision Classifier : ", ADT.score(x_test, y_test) * 100)

    print("\n----------------------------------------------------\n")

#=====================
#
# ENTRY POINT 
#
#=====================

def main():

    print("\n\t----- MNIST Case Study Using Boosting -----\n\t")
    
    Boosting_Using_DT()

#=====================
#
# CODE STARTER
#
#=====================

if __name__ == "__main__":
    main()
