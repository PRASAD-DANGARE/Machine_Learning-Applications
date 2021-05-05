'''
Classifier        :   User Defined K Nearest Neighbour
Dataset           :   Iris Dataset
Features          :   Sepal Width, Sepal Length, Petal Width, Petal Length
Labels            :   Versicolor, Setosa, Virginica 
Training Dataset  :   75 Entries
Testing Dataset   :   75 Entries
Author            :   Prasad Dangare
Function Name     :   CalculateDistance, Shortest, MarvellousKNN
Class Name        :   Marvellous

'''

# Mymodules.py File Contain Modules For KNN So We Have To Import It Into User_Defined_KNN.py File

from Mymodules import *

def CalculateDistance(X,Y):
    return distance.euclidean(X,Y)
    
class Marvellous():
    def fit(self,TrainingData, TrainingTarget):
        self.TrainingData = TrainingData
        self.TrainingTarget = TrainingTarget
          
    def predict(self,TestData):
        predictions = []
        for row in TestData:
            label = self.Shortest(row)
            predictions.append(label)
        return predictions
        
    def Shortest(self,row):
        Minindex = 0
        MinDistance = CalculateDistance(row,self.TrainingData[0])
        
        for i in range(1,len(self.TrainingData)):
            Distance = CalculateDistance(row,self.TrainingData[i])
            if Distance < MinDistance:
                MinDistance = Distance
                Minindex = i
        return self.TrainingTarget[Minindex]
        
def MarvellousKNN():
    Line = "*"*50
    iris = load_iris()      # 150 / 5 (columns) Total is 5 columns

    data = iris.data        # 150 / 4 (columns) seprate in 2 columns 4, 1
    target = iris.target    # 150 / 1 (columns)
    
    print(Line)
    print("Actual Dataset")
    print(Line)
    for i in range(len(iris.target)):
        print("ID : %d Feature : %s, Label : %s" %(i,iris.data[i], iris.target[i]))
#   75/4  (1)    75/4 (2)    75/1 (3)      75/1 (4)
    data_train, data_test, target_train, target_test = train_test_split(data,target,test_size = 0.5)
    
    print(Line)
    print("Training Data set")
    print(Line)
    for i in range(len(data_train)):
        print("ID : %d Feature : %s, Label : %s" %(i,data_train[i], target_train[i]))
            
    print(Line)
    print("Testing Data set")
    print(Line)
    for i in range(len(data_test)):
        print("ID : %d Feature : %s, Lebel : %s" %(i,data_test[i], target_test[i]))
    
    print(Line)
    mobj = Marvellous()     # Marvellous(5)
    
    mobj.fit(data_train, target_train)
  
    ret = mobj.predict(data_test)       #  2  75/4
    
    print("Result of Machine Learning Model")
    print(Line)
    for i in range(len(data_test)):
        print("ID : %d Expectation : %s, Prediction : %s" %(i, target_test[i],ret[i]))
    print(Line)

    icnt = 0
    for i in range(len(data_test)):
        if target_test[i] != ret[i]:
            icnt = icnt + 1
    print("Number of wrong answers by the ML model : ",icnt)
    print(Line)
    
    Accuracy = accuracy_score(target_test,ret)
    return Accuracy
    
def main():
    ret = MarvellousKNN()
    print("Accuracy of KNN is : ",ret*100,"%")
    print("*"*50)
    
if __name__ == "__main__":
    main()
