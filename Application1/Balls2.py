'''

Classifier        :   Decision Tree
Dataset           :   Balls Dataset
Features          :   Weight & Surface Type
Labels            :   Tennis And Cricket 
Training Dataset  :   15 Entries
Testing Dataset   :   1 Entry
Author            :   Prasad Dangare
Function Name     :   main

'''

from sklearn import tree

# Rough 1
# Smooth 0
# Tennis 1
# Cricket 2

def main():

    # Step 1 & 2 (get the data, clean , prepare, & manipulate the data)
    
    Features = [[35,1], [47,1], [90,0], [48,1], # creating list of list (1 is weight, 2 is surface)
               [90,0], [35,1], [92,0], [35,1],
               [35,1], [35,1], [96,0], [43,1],
               [110,0], [35,1], [95,0]]
        
    Label = [1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2] # labeled data (tennis, cricket ball) 

    # Step 3 (decide the ML Algorithm, train the dataset)
    
    dobj = tree.DecisionTreeClassifier()
        
    #Step 4 (test the algorithm with some test dataset)
    
    dobj.fit(Features, Label) # fit method use for training
        
    #Step 5 (depends on the test result improve the algorithm)
    
    result = dobj.predict([[40, 1]]) # predict method use for testing
        
    print("Ball is ", result)
        
if __name__ == "__main__":
    main()
