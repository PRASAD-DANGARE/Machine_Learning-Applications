'''

Classifier        :   Decision Tree
Dataset           :   Balls Dataset
Features          :   Weight & Surface Type
Labels            :   Tennis And Cricket 
Training Dataset  :   15 Entries
Testing Dataset   :   1 Entry
Author            :   Prasad Dangare
Function Name     :   Balls_Dataset

'''

from sklearn import tree

# Rough 1 
# Smooth 0 
# Tennis 1 
# Cricket 2

def Balls_Dataset(weight, surface):
    
    Features = [[35,1], [47,1], [90,0], [48,1],
               [90,0], [35,1], [92,0], [35,1],
               [35,1], [35,1], [96,0], [43,1],
               [110,0], [35,1], [95,0]]
    
    Labele = [1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2]
    
    dobj = tree.DecisionTreeClassifier()
    
    dobj.fit(Features,Labele)
    
    result = dobj.predict([[weight,surface]])
    
    if result == 1:
        print("\nYour object looks like Tennis ball")
    else:
        print("\nYour object looks like cricket ball")

def main():
    
    print("\n---------- Supervised Machine Learning -----------")
    print("\nEnter weight of object : ")
    
    weight = int(input())
    print("\nEnter surface type of object : ")
    surface = input()
    
    if surface.lower() == "rough": # convert into lower case
        surface = 1
    elif surface.lower() == "smooth":
        surface = 0
    else:
        print("\nInvalid input")
        return
        
    Balls_Dataset(weight, surface)
    
if __name__ == "__main__":
    main()
    
