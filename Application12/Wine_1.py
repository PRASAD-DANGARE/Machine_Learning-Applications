'''

Classifier : K Nearest Neighbour

DataSet    : Inbuilt Winne Predictor Dataset

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

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree

# ********************
# ML Operation
# ********************

def Wine_Predictor():

    wine = load_wine()

    #print(wine.data)
    
    print("\nTotal Volume Of Dataset..")
    print(wine.data.shape)
    print("\n")
    
    #print(wine.target)

    print("Grade Of Wines In Class..")
    print(wine.target_names)
    print("\n")
    
    print("13 Features Of Wines...")
    print(wine.feature_names)
    print("\n")

    Data = wine.data
    Target = wine.target
    
    Data_train, Data_test, Target_train, Target_test = train_test_split(Data, Target, test_size = 0.5)
    
    print("Data Split For Training Is 50%")
    print(Data_train.shape)
    print("\n")
    
    print("Data Split For Testing Is 50%")
    print(Data_test.shape)
    print("\n")

    #k = 7
    #cobj = KNeighborsClassifier(n_neighbors = k)

    cobj = tree.DecisionTreeClassifier()
    cobj.fit(Data_train, Target_train)

    output = cobj.predict(Data_test)

    Accuracy = accuracy_score(Target_test, output)

    print("Accuracy Using Decision Tree is : ", Accuracy*100, "%")

    #Test_Dataset = [[14.23,1.71,2.43,15.6,127,2.8,3.06,0.28,2.29,5.64,1.04,3.92,1065]] # step 3, 5 (class 0)
    #Test_Dataset = [[12.37,0.94,1.36,10.6,88,1.98,0.57,0.28,0.42,1.95,1.05,1.82,520]] # (class 1)
    
    Test_Dataset = [[12.86,1.35,2.32,18,122,1.51,1.25,0.21,0.94,4.1,0.76,1.29,630]] # (class 2)
    Result = cobj.predict(Test_Dataset)
    
    print("Wine Grade Is : ", wine.target_names [Result])
    print("\n")

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
