'''
Classifier        :   Linear Regression with User Defined Algorithm
Dataset           :   Head Brain Dataset
Features          :   Gender, Age, Head Size, Brain Weight
Labels            :   ----
Training Dataset  :   237
Testing Dataset   :   ----
Author            :   Prasad Dangare
Function Name     :   MeanData, HeadBrain

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def MeanData(arr):

    size = len(arr)
    sum = 0

    for i in range(size):
        sum = sum + arr[i]

    return (sum / size)

def HeadBrain(Name):
    dataset = pd.read_csv(Name)
    print("size of our data set is : ", dataset.shape)

    X = dataset["Head Size(cm^3)"].values
    Y = dataset["Brain Weight(grams)"].values

    print("Length of X ", len(X))
    print("Length of Y ", len(Y))

    Mean_X = MeanData(X)
    Mean_Y = MeanData(Y)

    print("Mean of Independent variables is ", Mean_X)
    print("Mean of Dependent variables is ", Mean_Y)

    # m = (Sum(X-XB)*(Y-YB)) / Sum(X-XB)^2

    numerator = 0
    denomenator = 0

    for i in range(len(X)):
        numerator = numerator + (X[i] - Mean_X) * (Y[i] - Mean_Y)
        denomenator = denomenator + (X[i] - Mean_X)**2

    m = numerator / denomenator
    print("Value of slope is ", m)

    # Y = MX + C
    # C = Y - MX
    # C = Mean_Y - (m * Mean_X)

    c = Mean_Y - (m * Mean_X)
    print("Value of Y intersect is ", c)

    X_Start = np.min(X) - 200
    X_End = np.max(X) + 200

    x = np.linspace(X_Start, X_End)
    y = m*x + c

    plt.plot(x,y,color = 'r', label = "Line of Regression")
    plt.scatter(X,Y, color = 'b', label = "Data Plot")

    plt.xlabel("Head Size")
    plt.ylabel("Brain Weight")
    plt.legend()
    plt.show()

    # R Square Calculation

    #print("Mean_x is ", MeanData(X))

def main():

    print("Enter file name of dataset : ")
    name = input()

    HeadBrain(name)

if __name__ == "__main__":
    main()
