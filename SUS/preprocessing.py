import numpy as np

# test train split for 1-dimentional feature vector
def test_train_split(X,y, sample_size=0.7,suffle=False):
    k = int(len(X)*sample_size)
    A = np.column_stack((X,y))  #stacking the feature and lables into single array A
    if suffle == True:
        np.random.shuffle(A)       #suffling data into random manner

    train = A[0:k,:]
    test = A[k:,:]
    return  test[:,:-1], test[:,-1].astype('int32'),train[:,:-1],train[:,-1].astype('int32')

def Standard_Scaler(X):
    A = []
    B = X.T
    for row in B:
        A.append((row-np.mean(row))/np.std(row))
    x = np.array(A).T
    return x