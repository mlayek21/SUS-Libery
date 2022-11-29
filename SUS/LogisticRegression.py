'''
Logistics Regression:
'''
import numpy as np
class LogisticRegression:
    # initilise the parameters
    def __init__(self, learning_rate=1e-2, n_iter=1000, ntol = 1e-5):
        self.n_iter = n_iter
        self.lr = learning_rate
        self.ntol = ntol
        self.w = None
        self.b = None
    
    # sigmoid function
    def sigmoid(self, X):
        z = np.dot(X,self.w)
        return (1/(1+np.exp(-z)))
    
    # fit methode
    def fit(self,x,y):
        X = np.hstack([np.ones((x.shape[0],1)),x])
        # y = y.shape()
        n_samples, n_features = X.shape
        self.w = np.ones(n_features)  #initilise the weights
        loss = []

        # gradient descent
        for i in range(self.n_iter):
            y_pred = self.sigmoid(X)
            loss.append((-np.sum(-y*np.log(y_pred)+(1-y)*np.log(1-y_pred)))/n_samples)
            #calculate gradient of weights and bias
            dw = np.dot(X.T,(y_pred-y))/n_samples
            #update weights
            self.w -= self.lr*dw
            if i>100 and np.linalg.norm(np.mean(loss[-100:])-loss[-1])<=self.ntol:
                print("Model converge in "+str(i)+" iteration")
                return loss
        return loss
    
    # predict methode
    def predict(self, x):
        X = np.hstack([np.ones((x.shape[0],1)),x])
        y = self.sigmoid(X)
        return np.where(y <= 0.5, 0, 1)