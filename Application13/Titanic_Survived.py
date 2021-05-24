'''

Classifier  :   Logistic Regression 
DataSet     :   TitanicDataset.csv
Features    :   Passenger id, Gender, Age, Fare, Class etc 
Labels      :   -
Author      :   Prasad Dangare
Function Name : Titanic_Logistic

'''

# ===================
# Imports
# ===================

import numpy as np                                              # use for multi-dimensional arrays.
import pandas as pd                                             # use for data cleaning, analysis, data frames
import seaborn as sb                                            # data visualization library based on matplotlib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression             # Importing LogisticRegression For Traning / Testing
from seaborn import countplot                                   # countplot Show the counts of observations in each categorical bin using bars
import matplotlib.pyplot as plt                                 # plotting library
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from matplotlib.pyplot import figure, show                      # figure()in pyplot module of matplotlib is used to create a new figure(object), 
                                                                #show()in pyplot module of matplotlib is used to display all the figures.

# ===================
# ML Operation
# ===================

def Titanic_Logistic(): 

    print("\nInside Titanic Logistic Function\n")

    # *************************
    # Step 1 : Load The Data
    # *************************

    Titanic_Data = pd.read_csv("TitanicDataset.csv")

    print("\nFirst 5 Entities From Loaded Data Set : \n")
    print(Titanic_Data.head()) # it shows first 5 records
    
    print("\nNumber Of Passengers Are : " + str(len(Titanic_Data)))

    # *************************
    # Step 2 : Analyse The Data
    # *************************

    print("\nVisulation : Survived And Non Survived Passengers : ")
    figure()
    target = "Survived"
    
    countplot(data = Titanic_Data, x = target).set_title("Survived Vs Non Survived") # data is the keyword argument
    show()

    print("\nVisulation : Survived And Non Survived Passengers Based On Gender : ")
    figure()
    target = "Survived"
    
    countplot(data = Titanic_Data, x = target, hue = "Sex").set_title("Survived And Non Survived Passengers Based On Gender")
    show()

    print("\nVisulation : Survived And Non Survived Passengers Based On Passenger Class : ")
    figure()
    target = "Survived"
    
    countplot(data = Titanic_Data, x = target, hue = "Pclass").set_title("Survived And Non Survived Passengers Based On Passenger Class")
    show()

    print("\nVisulation : Survived Vs Non Survived Based On Age : ")
    figure()
    
    Titanic_Data ["Age"].plot.hist().set_title("Survived Vs Non Survived Based On Age")
    show()

    print("\nVisulation : Survived Vs Non Survived Based On Fare : ")
    figure()
    
    Titanic_Data ["Fare"].plot.hist().set_title("Survived Vs Non Survived Based On Fare")
    show()

    # *************************************
    # Step 3 : Data Cleaning
    #        : Data Modification / Data Rangling
    # *************************************

    Titanic_Data.drop("zero", axis = 1, inplace = True) # drop method he column/row delete karti, inplace ha zaga war delete kala
    
    print("\nData After Column Removal Of zero : \n") 
    print(Titanic_Data.head(5)) 

    Sex = pd.get_dummies(Titanic_Data["Sex"])               # get_dummies it like label encoder, it split the column into 2 parts male, female,
    
    print("\nSex Column Classification As 0 And 1 : \n")    # get_dummies It converts categorical data into dummy
    print(Sex.head(5))                                      # get_dumies is use to clean the data

    Sex = pd.get_dummies(Titanic_Data["Sex"], drop_first = True) # we remove the female column 
    
    print("\nSex Column After Removing Female Column : \n")
    print(Sex.head(5))

    Pclass = pd.get_dummies(Titanic_Data["Pclass"])         # it create 3 dummies as 1, 2, 3
    
    print("\nPassenger Class Classification In 1,2,3 : \n")
    print(Pclass.head(5))

    Pclass = pd.get_dummies(Titanic_Data["Pclass"], drop_first = True)
    
    print("\nAfter First Class Passenger Column Removal : \n")
    print(Pclass.head(5))

    # Concat Sex And Pclass Field In Our Data Set

    Titanic_Data = pd.concat([Titanic_Data, Sex, Pclass], axis = 1) 
    
    print("\nData After Concination Sex And Pclass : \n")
    print(Titanic_Data.head(5))

    # Giving New Name To Concatenate Fields

    Titanic_Data.rename(columns = {Titanic_Data.columns[9]: "New Sex" }, inplace = True) 
    Titanic_Data.rename(columns = {Titanic_Data.columns[10]: "2 Class" }, inplace = True) 
    Titanic_Data.rename(columns = {Titanic_Data.columns[11]: "3 Class" }, inplace = True) 
    
    print("\nData After Updation Of Names : \n")
    print(Titanic_Data.head())

    # Removing Un Necessary Fields

    Titanic_Data.drop(["Sex", "sibsp", "Parch", "Embarked"], axis = 1, inplace = True) # axis = 1 means drop the column
    
    print("\nData After Removal Of Columns Sex, sibsp, Parch, Embarked : \n")
    print(Titanic_Data.head(5))

    # Divide The Data Set Into x And y

    x = Titanic_Data.drop("Survived", axis = 1)
    y = Titanic_Data["Survived"]

    # Split The Data For Traning And Testing Purpose

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5)

    obj = LogisticRegression(max_iter = 1000) # from max_iter we can increase the ITERATIONS LIMIT

    
    # *****************************
    # Step 4 : Train The Data Set
    # *****************************

    obj.fit(x_train, y_train)

    # ***********************************
    # Step 5 : Test / Train The Data Set
    # ***********************************

    output = obj.predict(x_test)
    
    print("\nAccuracy Of Given Data Set is : \n")
    print(accuracy_score(y_test, output))
    
    print("\nConfusion Matrix Is : \n")
    print(confusion_matrix(y_test, output))

# =======================
# Entry Point
# =======================

def main():

    print("_____ Titanic Logistic Case Study _____")
    print("___ Using Logistic Regression ___")
    
    Titanic_Logistic()

# ===================
# Starter
# ===================

if __name__ == "__main__":
    main()
