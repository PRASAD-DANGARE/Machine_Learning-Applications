'''

Classifier        :   Decision Tree
Dataset           :   Iris Dataset
Features          :   Sepal Width, Sepal Length, Petal Width, Petal Length
Labels            :   Versicolor, Setosa, Virginica 
Training Dataset  :   147 Entries
Testing Dataset   :   3 Entries
Author            :   Prasad Dangare
Function Name     :   main

'''

from sklearn.datasets import load_iris
import numpy as np
from sklearn import tree

def main():

    dataset = load_iris()

    print("Features of dataset")
    print(dataset.feature_names)
    print("Target names of dataset")
    print(dataset.target_names)

    # Indices of removed elements
    index = [1, 51, 101]

    # Training data with removed elements
    train_target = np.delete(dataset.target, index)
    train_feature = np.delete(dataset.data, index, axis = 0)

    # Testing data for testing on trainning data
    test_target = dataset.target[index]
    test_feature = dataset.data[index]

    # form dcision tree classifier
    obj = tree.DecisionTreeClassifier()

    # apply training data to form tree
    obj.fit(train_feature, train_target)

    # apply testing on trained data
    result = obj.predict(test_feature)

    print("result prediction by ML ", result)
    print("result expected ", test_target)

if __name__ == "__main__":
    main()
