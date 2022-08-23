import numpy as np
'''
Linear Regression Class:
'''
class LinearRegression:
    # initilising the parameters
    def __init__(self, learning_rate=1e-5, tolarence=1e-3):
        self.lr = learning_rate
        self.tol = tolarence
        self.w = None

    # fit function
    def fit(self, X, y):
        X = np.vstack((np.ones(X.shape[0], ), X.T)).T  # adding bias to the feature vector
        np.random.seed(0)  # random seed
        self.w = np.random.rand(X.shape[1])  # initilizing the weight vector
        N = len(X)
        i = 1

        while True:
            Y = np.dot(X, self.w.T)
            w_grad = (1 / N) * np.dot(X.T, (Y - y))  # gradient of the weight vector
            self.w = self.w - (self.lr * w_grad)
            i += 1
            if np.all(
                    np.abs(w_grad) <= self.tol):  # if step value enter with in the range of tolerance then look break or
                break
        return self.w

    # predict function
    def predict(self, x):
        x = np.vstack((np.ones(x.shape[0], ), x.T)).T
        return np.dot(x, self.w)

'''
Polynomial Regression Class:
'''
class Polynomial_Regression:
    
    #initilized the parameters
    def __init__(self,degree, learning_rate, lambda_, tolerance):
        self.n = degree       #degree of the polynomial
        self.l = learning_rate    #learning rate
        self.r = lambda_
        self.tol_ = tolerance
        self.w = None
    
    #function for converting the feature vector into polynomial matrix 
    def poly_matrix(self,x):
        A = []              # blank list of Polynomial Matrix of the feature vector
        for row in x:
            for i in range(self.n+1):
                A.append(row**i) 
        A = np.ravel(A)
        return A.reshape(len(x),(self.n+1))
    
    #Intercept function
    def intercept(self):
        a = self.w[0]
        return a
   
    #weights vector after fiting the model
    def weights(self):
        w = self.w[(-self.w.shape[0]+1):]
        return w
    
    #function for train the model
    def  fit(self,x, y):
        np.random.seed(0)
        X = self.poly_matrix(x)
        self.w = np.random.randn(X.shape[1])    # initilising the weights vectors
        N = len(X)
        i = 1
        
        while True: 
            y_pred = np.dot(X, self.w)
            
            #gradient of the weight vector
            w_grad = (1/N) * np.dot(X.T, (y_pred -y)) + (( self.r * self.w )  / N) 
            
            #updating the gradient to a new current value
            self.w = self.w - (self.l * w_grad)
            #condition for convergence
            if np.all(np.abs(w_grad) <= self.tol_): #if step value enter with in the range of tolerance  
                break
            i+=1
            
        return 
    #function for predict the output for the given input    
    def predict(self,x):
        X = self.poly_matrix(x)
        return np.dot(X,self.w) 


'''
Linear Regression using Gaussian basis function:
'''
class GBF_linear_regression:
    
    #initilising the parameters
    def __init__(self, num_basis, sigma):
        self.n_basis = num_basis          #number of basis function
        self.sd = sigma                  #standered deviation of gaussian
        self.basis = None               #mean of the gaussian
        self.w = None                  #weights
        
    #Probablity density function for Normal Distribution
    def GBF(self,x,mu):
        pdf = np.exp((-0.5)*((x-mu)/self.sd)**2) #probably density function for the normal distribution
        return pdf
    
    #weights vector after fiting the model
    def weights(self):
        return np.ravel(self.w)
    
    #function to randomly choose the mean for n gaussians
    def basis_f(self,x):
       #randomly seed
        np.random.seed(10)
        basis_idx = np.random.choice(x.shape[0], self.n_basis, replace = False)  #selecting index row from the feature matrix
        self.basis = np.array(x[basis_idx])                                     #assign the selected row into the mean of that gaussian
        return
    
    #modle fit methode
    def fit(self,x,y):
        self.basis_f(x)     #generating mu
        phi = np.asmatrix(np.ones((x.shape[0], 1)))  #initilising the bias
        
        #itterating and generating the phi matrix for each basis and features
        for mu in self.basis:
            col = np.array([self.GBF(obs,mu) for obs in np.array(x[:])]) # calculating pdf for each feature values for given basis mu 
            col = col.reshape(col.shape[0],1)                            
            phi = np.hstack((phi,col))                                      # adding the bias term
        self.w = ((phi.T * phi)**-1)*phi.T*y.reshape(x.shape[0],1)    # calucate the weights

        return 
    
    #predice methode
    def predict(self,x):
        t = np.asmatrix(np.ones((x.shape[0],1)))
        for mu in self.basis:
            col = np.array([self.GBF(obs,mu)for obs in x])
            col = col.reshape(col.shape[0],1)
            t = np.hstack((t,col))
        return np.ravel(t*self.w)


