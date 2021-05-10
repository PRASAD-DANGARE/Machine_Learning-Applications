'''
Classifier        :   Linear Regression
Dataset           :   Head Brain Dataset
Features          :   Gender, Age, Head Size, Brain Weight
Labels            :   ----
Training Dataset  :   237
Testing Dataset   :   ----
Author            :   Prasad Dangare
Function Name     :   MarvellousHeadBrainPredictor

'''

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def MarvellousHeadBrainPredictor():

    # Load data
    data = pd.read_csv('HeadBrain.csv')
    print("Size Of DataSet ", data.shape)

    X = data['Head Size(cm^3)'].values
    Y = data['Brain Weight(grams)'].values

    X = X.reshape((-1,1))
    n = len(X)

    reg = LinearRegression()
    
    reg = reg.fit(X,Y)
    y_pred = reg.predict(X)
    r2 = reg.score(X,Y)
    print("Value of R Square is", r2)

def main():

    print("------Supervised Machine Learning------")
    print("Linear Regression On Head Brain Size DataSet")

    MarvellousHeadBrainPredictor()

if __name__ == "__main__":
    main()
