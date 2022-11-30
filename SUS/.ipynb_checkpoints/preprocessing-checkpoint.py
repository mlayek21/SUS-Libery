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

class Minmax:
    def __init__(self):
        self.max= None
        self.min = None
        
    def fit(self,X):
        self.max = np.max(X,axis=0)
        self.min = np.min(X,axis=0)
        return
        
    def fit_transform(self,X):
        self.max = np.max(X,axis=0)
        self.min = np.min(X,axis=0)
        for i in range(X.shape[1]):
            X[:,i] = (X[:,i]-self.min[i])/(self.max[i]-self.min[i])
        return X   
    def transform(self,X):
        for i in range(X.shape[1]):
            X[:,i] = (X[:,i]-self.min[i])/(self.max[i]-self.min[i])
        return X  
    
class Standard_Scaler:
    def __init__(self):
        self.mean= None
        self.std = None
        
    def fit(self,X):
        self.mean = np.mean(X,axis=0)
        self.std = np.std(X,axis=0)
        return
        
    def fit_transform(self,X):
        self.mean = np.mean(X,axis=0)
        self.std = np.std(X,axis=0)
        for i in range(X.shape[1]):
            X[:,i] = (X[:,i]-self.mean[i])/(self.std[i])
        return X   
    def transform(self,X):
        for i in range(X.shape[1]):
            X[:,i] = (X[:,i]-self.mean[i])/(self.std[i])
        return X               

class PCA:
    
    def __init__(self, n_comp):
        self.n_comp = n_comp
        self.comp = None
        self.mean = None
        
    def fit (self, X):
        #mean
        self.mean = np.mean(X, axis = 0)
        X = X - self.mean
        
        #covarience
        cov = np.cov(X.T)
        
        #eigenvectors, eigen values
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        # v[:,i]
        
        # sort the eigen vectors
        eigenvectors = eigenvectors.T
        indx = np.argsort(eigenvalues)[::-1]
        
        eigenvalues = eigenvalues[indx]
        eigenvectors = eigenvectors[indx]
        
        #store the first n eigenvectors 
        self.comp = eigenvectors[0:self.n_comp]
        return self.comp, self.mean
    
    def transform(self, X):
        #project data
        X = X - self.mean
        return np.dot(X, self.comp.T)