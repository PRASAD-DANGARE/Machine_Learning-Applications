'''

Classifier  :   Logistic Regression
DataSet     :   diabetes.csv
Features    :   Glucose, BMI etc
Labels      :   ----
Author      :   Prasad Dangare
Date        :   30 June 2021

'''

#========================
#
# Imports
#
#========================

import pandas as pd 
from matplotlib.pyplot import figure, show  
from seaborn import countplot  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from warnings import simplefilter 

#========================
#
# Accuracy Operations
#
#========================

def Diabetes_LogisticRegression():

    simplefilter(action = 'ignore', category = FutureWarning) 

    print("\n\t----- Diabetes predictor using Logistic Regression -----\n\t")

    diabetes = pd.read_csv('diabetes.csv') 

    print("Columns Of Dataset : \n") 
    print(diabetes.columns)

    print("\nFirst 5 Records Of Dataset : \n") 
    print(diabetes.head())

    print("\n**************************************************************************\n")

    print("\nDimension / Volume Of Diabetes Data : {}\n".format(diabetes.shape))

    X_train, X_test, y_train, y_test = train_test_split(diabetes.loc[:, diabetes.columns != 'Outcome'], 
    diabetes['Outcome'], stratify = diabetes['Outcome'], random_state = 64)

    print("\n-------------------------------------------------------------------\n")

    LogR = LogisticRegression(max_iter = 1000)
    LogR.fit(X_train, y_train)

    print("Training Set Accuracy : {:.3f}\n".format(LogR.score(X_train, y_train) * 100)) 
    print("Testing Set Accuracy : {:.3f}\n".format(LogR.score(X_test, y_test) * 100))

    print("\n-------------------------------------------------------------------\n")

    LogR001 = LogisticRegression(C = 0.01, max_iter = 1000)
    LogR001.fit(X_train, y_train) 

    print("Training Set Accuracy : {:.3f}\n".format(LogR001.score(X_train, y_train) * 100)) 
    print("Testing Set Accuracy : {:.3f}\n".format(LogR001.score(X_test, y_test) * 100))

    print("\n-------------------------------------------------------------------\n")

    print("\nVisulation : If Diabetes Detected (1) And If Not Detected (0)\n")
    figure()
    target = "Outcome"
    countplot(data = diabetes, x = target).set_title("Visulation Of Diabetes Detected And Non Detected") 
    show()

#========================
#
# Entry Point 
#
#========================

def main():

    Diabetes_LogisticRegression()

#========================
#
# Code Starter
#
#========================

if __name__ == "__main__":
    main()



