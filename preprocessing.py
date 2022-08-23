import numpy as np
class preprocessing:
    def __init__(self) -> None:
        self.k = None

    # test train split for 1-dimentional feature vector
    def test_train_split(self,X,y, k,suffle=False):
        self.k = k
        A = np.column_stack((X,y))  #stacking the feature and lables into single array A
        if suffle == True:
            np.random.shuffle(A)       #suffling data into random manner
            
        n = len(x)//self.k
        train = A[0:k,:]
        test = A[k:,:]
        return  train[:,:-1],train[:,-1],test[:,:-1], test[:,-1]