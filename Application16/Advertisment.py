'''

Function Name :  Advertisment_Predictor, Data_Splitting, Dtype_Change
Classifier    :  Linear Regression 
DataSet       :  advertising.csv
Features      :  TV, RADIO, NEWSPAPER
Labels        :  SALES
Author        :  Prasad Dangare
Date          :  21 July 2021

'''

# ===================
#
# Imports
#
# ===================

import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression

# ===================
#
# ML Operation
#
# ===================

def Advertisment_Predictor(train_x, train_y, test_x, test_y):

    obj = LinearRegression()
    obj.fit(train_x, train_y)
    print("R Square value is : ", obj.score(train_x, train_y) * 100)

def Data_Splitting(data):

    length = len(data)
    train_size = int(0.8 * length)
    test_size = length - train_size
    train_data = data.head(train_size)
    test_data = data.tail(test_size)
    return train_data, test_data

def Dtype_Change(data):
    return data.astype(np.int64)

# =======================
#
# Entry Point
#
# =======================


def main():

    path = "Advertising.csv"
    data = pd.read_csv(path)

    print("\n")
    print(data.info())
    
    attributes = ['TV','radio','newspaper']
    data_label = Dtype_Change(pd.read_csv(path, usecols = attributes))
    data_target = Dtype_Change(pd.read_csv(path, usecols = ['sales']))
    train_label, test_label = Data_Splitting(data_label)
    
    train_target, test_target = Data_Splitting(data_target)
    print("\n")
    print(train_label.info())
    print("\n")
    print(train_target.info())
    print("\n")
    Advertisment_Predictor(train_label, train_target, test_label, test_target)

# =======================
#
# Code Starter
#
# =======================


if __name__=="__main__":
    main()
