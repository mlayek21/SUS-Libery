import numpy as np

class LinearRegression:
    def __init__(self,kernal="linear", degree = None, sigma = 0.1):
        self.kernal = kernal
        self.w = None 
        self.m = None
        self.n = None
        self.d = degree
        self.sd = sigma
            
    def check(self,X):
        if (self.kernal == "linear"):
            X = np.hstack([np.ones((self.m,1)),X])       
        elif self.kernal == "RBF":
            X = self.rbf(X)
        elif self.kernal == "polynomial":
            X = self.poly(X)
        return X
    
    def basis_f(self,x):
       #randomly seed
        np.random.seed(10)
        basis_idx = np.random.choice(x.shape[0], self.d, replace = False)  #selecting index row from the feature matrix
        self.basis = np.array(x[basis_idx])                                     #assign the selected row into the mean of that gaussian
        return
    
    def GBF(self,x,mu):
        pdf = np.exp((-0.5)*((x-mu)/self.sd)**2) #probably density function for the normal distribution
        return pdf
    
    def rbf(self,x):
        self.basis_f(x)
        phi = np.asmatrix(np.ones((x.shape[0], 1)))  #initilising the bias
        
        #itterating and generating the phi matrix for each basis and features
        for mu in self.basis:
            col = np.array([self.GBF(obs,mu) for obs in np.array(x[:])]) # calculating pdf for each feature values for given basis mu 
            col = col.reshape(col.shape[0],1)                            
            phi = np.hstack((phi,col))
        return phi
    
    def poly(self,x):
        if self.d == None:
            print("Enter the degree of polynomial kernal")
            self.d = int(input())
        A = []              # blank list of Polynomial Matrix of the feature vector
        for row in x:
            for i in range(self.d+1):
                A.append(row**i) 
        A = np.ravel(A)
        return A.reshape(len(x),(self.d+1))
 
    def fit(self,X,y):
        self.m,self.n = X.shape
        X = self.check(X)
        self.w = np.linalg.inv(X.T@X)@X.T@y
        return self.w

    def predict(self,x):
        x = self.check(x)
        return self.w@x.T