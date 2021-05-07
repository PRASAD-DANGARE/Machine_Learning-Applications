'''
Classifier        :   Linear Regression Using Least Square Method 
Formula           :   1) Equation Of Line : Y = mX + C, 2) Rsquare : [sum(yp-yb)^2] / [sum(y-yb)^2]
Dataset           :   Play Predictor Dataset
Features          :   ----
Labels            :   ----
Training Dataset  :   ----
Testing Dataset   :   ----
Author            :   Prasad Dangare
Function Name     :   MarvellousPredictor

'''

import pandas as pd
import numpy as np

def MarvellousPredictor(path):

    data = pd.read_csv(path)
    print("Dataset loaded successfully with the size ", len(data))

    X = [1,2,3,4,5]
    Y = [3,4,2,4,5]

    X_Mean = np.mean(X)
    Y_Mean = np.mean(Y)

    Numerator = 0
    Denomenator = 0
    RSquare = 0

    for i in range(len(X)):
        Numerator = Numerator + ((X[i] - X_Mean)* (Y[i] - Y_Mean))
        Denomenator = Denomenator + ((X[i] - X_Mean)**2)

    m = Numerator / Denomenator

    print("Values of X :", X)
    print("Values of Y :", Y)

    print("Values of m :", m)

    # Y = mX + C

    c = Y_Mean - (m * X_Mean)

    print("Values of c :", c)

    Numerator = 0
    Denomenator = 0

    # [sum(yp-yb)^2] / [sum(y-yb)^2] formula for Rsquare
    
    for i in range(len(X)): 
        Numerator = Numerator + (((m*X[i] + c) - Y_Mean)**2)
        Denomenator = Denomenator + ((Y[i] - Y_Mean)**2)

    RSquare = Numerator / Denomenator

    print("Value of RSquare is : ", RSquare)

def main():

    print("_____Marvellous Play Predictor_____")
    print("Enter the path of the file which contains dataset : ")
    path = input()

    MarvellousPredictor(path)

if __name__ == "__main__":
    main()
