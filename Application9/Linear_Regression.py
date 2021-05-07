'''
Classifier        :   Linear Regression With User Defined Algorithm 
Formula           :   1) least Square Method, 2) Equation of line, 3) R Square
Dataset           :   ----
Features          :   ----
Labels            :   ----
Training Dataset  :   ----
Testing Dataset   :   ----
Author            :   Prasad Dangare
Function Name     :   MarvellousPredictor
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def MarvellousPredictor():

    # load data
    X = [1,2,3,4,5]
    Y = [3,4,2,4,5]

    print("Values Of Independent Of Variables x", X)
    print("Values Of Dependent Of Variables y", Y)

    # least Square Method

    mean_x = np.mean(X)
    mean_y = np.mean(Y)

    print("Mean Of Independent Variables x", mean_x)
    print("Mean Of Dependent Variables y", mean_y)
    n = len(X)

    numerator = 0
    denomenator = 0

    # Equation of line is y = mx + c

    for i in range(n):
        numerator += (X[i] - mean_x) * (Y[i] - mean_y)
        denomenator += (X[i] - mean_x)**2

    m = numerator / denomenator

    # c = y' - mx'

    c = mean_y - (m * mean_x)

    print("Slope of Regression line is ", m) # 0.4
    print("Y Intercept of Regression line is ", c) # 2.4

    # Display plotting of above points

    x = np.linspace(1,6,n)

    y = c + m * x

    plt.plot(x,y, color = '#58b970', label = 'Regression Line')
    plt.scatter(X,Y, color = '#ef5423', label = 'Scatter Plot')

    plt.xlabel('X - Independent Variables')
    plt.ylabel('Y - Dependent Variables ')

    plt.legend()
    plt.show()

    # findout goodness of fit ie R Square

    ss_t = 0
    ss_r = 0

    for i in range(n):
        y_pred = c + m * X[i]
        ss_t += (Y[i] - mean_y) ** 2
        ss_r += (Y[i] - y_pred) ** 2

    r2 = 1 - (ss_r/ss_t)
    print("Goodness Of fit using R2 Method is ", r2)

def main():

    print("-----Supervised Machine Learning-----")
    print("Linear Regression")

    MarvellousPredictor()

if __name__ == "__main__":
    main()

