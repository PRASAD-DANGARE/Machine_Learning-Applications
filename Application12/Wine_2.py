'''

Classifier : K Nearest Neighbour

DataSet    : WinePredictor.csv

Features   : Alcohol, Malic acid , Ash, Alcalinity of ash , Magnesium ,Total phenols , Flavanoids ,
           : Nonflavanoid phenols , Proanthocyanins , Color intensity, Hue , OD280/OD315 of diluted wines, Proline.

Labels     : Class 1, Class 2, Class 3

Training Dataset    : 70% of 178 Entries

Testing Dataset     : 30% of 178 Entries

Author            :   Prasad Dangare

Function Name     :   Wine_Predictor

'''

# ********************
# Imports
# ********************

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ********************
# ML Operation
# ********************

def Wine_Predictor():

    Space = "*"*50

    data = pd.read_csv("WinePredictor.csv")

    print("\n",Space)
    print("\nVolume Of Dataset is : ", len(data), "And Features Are 13")

    print("\n",data)
    print("\n",Space)

    Target = data.Class

    Data_train, Data_test, Target_train, Target_test = train_test_split(data, Target, test_size = 0.3)

    print("\nData Split For Training Is : ")
    print("\n", Data_train)
    print("\n",Space)

    print("\nData Split For Testing Is : ")
    print("\n", Data_test)
    print("\n", Space)

    cobj = KNeighborsClassifier()
    cobj.fit(Data_train, Target_train)

    output = cobj.predict(Data_test)

    Accuracy = accuracy_score(Target_test, output)

    print("\nAccuracy Using KNN is : ", Accuracy*100, "%")
    print("\n", Space)

# ********************
# Entry Point
# ********************

def main():

    print("\t\t__________________Wine Predictor___________________")
    
    Wine_Predictor()

# ********************
# Starter
# ********************

if __name__ == "__main__":
    main()


