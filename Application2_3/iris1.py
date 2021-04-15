'''

Classifier        :   Decision Tree
Dataset           :   Iris Dataset
Features          :   Sepal Width, Sepal Length, Petal Width, Petal Length
Labels            :   Versicolor, Setosa, Virginica 
Training Dataset  :   150 Entries
Testing Dataset   :   ----
Author            :   Prasad Dangare
Function Name     :   ----

'''

from sklearn.datasets import load_iris

iris = load_iris()

print("Features name of iris dataset")
print(iris.feature_names)

print("Target name of iris dataset")
print(iris.target_names)

print("All Elements from iris dataset")

for i in range(len(iris.target)):
    print("ID : %d, Label %s, Features : %s" % (i, iris.data[i], iris.target[i]))
