'''

Classifier  :   K Nearest Neighbores
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
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split

#========================
#
# Accuracy Operations
#
#========================

def Diabetes_KNN():

    print("\n\t----- Diabetes predictor using K Nearest neighbour -----\n\t") 

    diabetes = pd.read_csv('diabetes.csv')

    print("Columns Of Dataset : \n") 
    print(diabetes.columns)

    print("\nFirst 5 Records Of Dataset : \n") 
    print(diabetes.head()) 

    print("\n**************************************************************************\n")

    print("\nDimension / Volume Of Diabetes Data : {}\n".format(diabetes.shape)) 

    X_train, X_test, y_train, y_test = train_test_split(diabetes.loc[:, diabetes.columns != 'Outcome'], 
    diabetes['Outcome'], stratify = diabetes['Outcome'], random_state = 66)

    print("\n-------------------------------------------------------------------\n")

    KNN = KNeighborsClassifier(n_neighbors = 8)

    KNN.fit(X_train, y_train)

    print('\nAccuracy Of K-NN Classifier On Traning Set : {:.2f}\n'.format(KNN.score(X_train, y_train) * 100))

    print('\nAccuracy Of K-NN Classifier On Testing Set : {:.2f}\n'.format(KNN.score(X_test, y_test) * 100))

    print("\n-------------------------------------------------------------------\n")

    Training_Accuracy = [] 
    Testing_Accuracy = []

    # Trying n_neighbors from 1 to 12 
    Neighbors_Settings = range(1, 12) 

    for N_neighbors in Neighbors_Settings: 

        # Build The Model 
        KNN = KNeighborsClassifier(n_neighbors = N_neighbors) 
        KNN.fit(X_train, y_train) 

        # Record Training Set Accuracy 
        Training_Accuracy.append(KNN.score(X_train, y_train)) 

        # Record Testing Set Accuracy 
        Testing_Accuracy.append(KNN.score(X_test, y_test)) 

    plt.plot(Neighbors_Settings, Training_Accuracy, label = "Training Accuracy") 
    plt.plot(Neighbors_Settings, Testing_Accuracy, label = "Testing Accuracy") 

    plt.ylabel("Accuracy") 
    plt.xlabel("n_neighbors") 
    plt.legend() 
    plt.savefig('KNN_Compare_Model') 
    plt.show() 

#========================
#
# Entry Point Function
#
#========================

def main():

    Diabetes_KNN()

#========================
#
# Code Starter
#
#========================

if __name__ == "__main__":
    main()

