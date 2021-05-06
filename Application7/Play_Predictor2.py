'''
Classifier        :   K Nearest Neighbour
Dataset           :   Play Predictor Dataset
Features          :   Whether, Temperature
Labels            :   Yes, No 
Training Dataset  :   30 Entries
Testing Dataset   :   1 Entry
Author            :   Prasad Dangare
Function Name     :   MarvellousPredictor

'''

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

def MarvellousPredictor(path):

    # Step 1
    data = pd.read_csv(path)
    print("Dataset loaded successfully with the size ", len(data))

    # Step 2
    Feature_name = ["Whether", "Temprature"]
    print("Feature Name Are ", Feature_name)

    Whether = data.Wether # names according to csv file
    Temprature = data.Temperature
    Play = data.Play

    lobj = preprocessing.LabelEncoder()

    WhetherX = lobj.fit_transform(Whether)
    TempratureX = lobj.fit_transform(Temprature)
    Label = lobj.fit_transform(Play)

    print("Encoded Whether is ")
    print(WhetherX)

    print("Encoded Temprature is ")
    print(TempratureX)

    features = list(zip(WhetherX, TempratureX))

    # Step 3
    obj = KNeighborsClassifier(n_neighbors = 3)
    obj.fit(features,Label)

    # Step 4
    output = obj.predict([[0,2]])

    if output == 1:
        print("You Can Play")
    else:
        print("Dont Play")

def main():

    print("_____Marvellous Play Predictor_____")
    print("Enter the path of the file which contains dataset : ")
    path = input()

    MarvellousPredictor(path)

if __name__ == "__main__":
    main()

