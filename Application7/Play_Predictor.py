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

def MarvellousPredictor(data_path):

    # step 1 Load data
    data = pd.read_csv(data_path, index_col = 0)
    print("Dataset loaded successfully with the size ", len(data))

    # step 2 Clean, prepare and manipulate data
    feature_names = ['Whether', 'Temprature']
    print("Features Name are : ", feature_names)

    whether = data.Wether
    Temperature = data.Temperature
    play = data.Play

    # Creating LabelEncoder
    LE = preprocessing.LabelEncoder()
    
    # Converting String Labels Into Numbers
    weather_encoded = LE.fit_transform(whether)
    print(weather_encoded)

    # Converting String Labels Into Numbers
    Temperature_encoded = LE.fit_transform(Temperature)
    Label = LE.fit_transform(play)
    print(Temperature_encoded)

    # combining weather and temprature into single list of tuples
    features = list(zip(weather_encoded,Temperature_encoded))

    # step 3 tarin data
    model = KNeighborsClassifier(n_neighbors = 3)
    
    # train the model using the traning sets
    model.fit(features,Label)

    # step 4 test data
    predicted = model.predict([[0,2]]) # 0 is overcast, 2 is mild
    print(predicted)

def main():

    print("_____Marvellous Play Predictor_____")
    print("Machine Learning Application")
    print("Play Predictor Application Using K Nearest Nighbour")

    MarvellousPredictor("PlayPredictor.csv")

if __name__ == "__main__":
    main()
