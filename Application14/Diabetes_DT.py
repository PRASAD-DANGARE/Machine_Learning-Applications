'''

Classifier  :   Decision Tree
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
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import dia
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

#========================
#
# Accuracy Operations
#
#========================

print("\n\t----- Diabetes Predictor Using Decision Tree -----\n\t")

diabetes = pd.read_csv('diabetes.csv')

print("Columns Of Dataset : \n")
print(diabetes.columns)

print("\nFirst 5 Records Of Dataset : \n")
print(diabetes.head())

print("\n**************************************************************************\n")

print("\nDimension / Volume Of Diabetes Data : {}\n".format(diabetes.shape))

X_train, X_test, y_train, y_test = train_test_split(diabetes.loc[:, diabetes.columns != 'Outcome'], 
diabetes['Outcome'], stratify = diabetes['Outcome'], random_state = 66)

DT = DecisionTreeClassifier(random_state = 0)
DT.fit(X_train, y_train)

print("\n-------------------------------------------------------------------\n")

print("Accuracy On Training Set With No Maximum Dept : {:.3f}\n".format(DT.score(X_train, y_train) * 100))

print("Accuracy On Testing Set : {:.3f}\n".format(DT.score(X_test, y_test) * 100))

print("\n-------------------------------------------------------------------\n")

DT = DecisionTreeClassifier(max_depth = 4, random_state = 0) 
DT.fit(X_train, y_train)

print("Accuracy On Training Set With Maximum Dept 4 Is : {:.3f}\n".format(DT.score(X_train, y_train) * 100)) 
print("Accuracy On Testing Set : {:.3f}\n".format(DT.score(X_test, y_test) * 100))

print("\n**************************************************************************\n")

print("Feature Importances : \n{}\n".format(DT.feature_importances_ * 100))


#===========================
#
# Graphical Representation
#
#===========================

def plot_feature_importances_diabetes(model): 
    
    plt.figure(figsize = (8, 6)) 

    n_features = 8 
    
    plt.barh(range(n_features), model.feature_importances_, align = 'center') 
    diabetes_features = [x for i, x in enumerate(diabetes.columns) if i != 8] 
    plt.yticks(np.arange(n_features), diabetes_features) 
    
    plt.xlabel("Feature Importance") 
    plt.ylabel("Feature") 
    plt.savefig('DT_Compare_Model') 
    
    plt.ylim(-1, n_features) 
    
    plt.show() 

plot_feature_importances_diabetes(DT)