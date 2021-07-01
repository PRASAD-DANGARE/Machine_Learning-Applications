'''
Classifier          :   Decision Tree
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

#=====================
#
# Accuracy Operation
#
#=====================

def MNIST_Using_DT():

    data = pd.read_csv('mnist.csv')

    DF_x = data.iloc[:,1:]  # Labels
    DF_y = data.iloc[:,0]   # Pixels

    x_train, x_test, y_train, y_test = train_test_split(DF_x, DF_y, test_size = 0.3, random_state = 5)

    #descision tree

    DT = DecisionTreeClassifier()
    DT.fit(x_train, y_train)

    print("\n----------------------------------------------------\n")

    print("Training Accuracy Using Decision Tree Classifier : ", DT.score(x_train, y_train) * 100)
    print("Testing Accuracy Using Decision Tree Classifier : ", DT.score(x_test, y_test) * 100)

    print("\n----------------------------------------------------\n")

#=====================
#
# ENTRY POINT 
#
#=====================

def main():

    print("\n\t----- MNIST Case Study Using Decision Tree -----\n\t")

    MNIST_Using_DT()

#=====================
#
# CODE STARTER
#
#=====================

if __name__ == "__main__":
    main()
