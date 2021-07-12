'''
Classifier          :   Heterogeneous Algorithm Technique
Dataset             :   Iris Dataset
Features            :   Sepal Width, Sepal Length, Petal Width, Petal Length
Labels              :   Versicolor, Setosa, Virginica 

Training Dataset    :   70% of 150 Entries
Testing Dataset     :   30% of 150 Entries

Author              :   Prasad Dangare
Date                :   12 July 2021

Function Name       :   Boosting

'''

#====================
#
# IMPORTS
#
#====================

from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import VotingClassifier 
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score

#=====================
#
# Accuracy Operation
#
#=====================

def Boosting():

    iris = load_iris() 

    x = iris['data'] 
    y = iris['target']

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 42, train_size = 0.3) 

    lR_clf = LogisticRegression() 
    RF_clf = RandomForestClassifier() 
    KNN_clf = KNeighborsClassifier()

    vot_clf = VotingClassifier(estimators = [('lr', lR_clf), ('rnd', RF_clf), ('knn', KNN_clf)], voting = 'hard') 

    vot_clf.fit(x_train, y_train) 

    pred = vot_clf.predict(x_test) 

    print("Testing accuracy is : ", accuracy_score(y_test, pred) * 100)


#=====================
#
# ENTRY POINT 
#
#=====================


def main():

    print("\n\t----- IRIS Case Study Using Heterogeneous Algorithm Technique -----\n\t")

    Boosting()

#=====================
#
# CODE STARTER
#
#=====================

if __name__ == "__main__":
    main()
