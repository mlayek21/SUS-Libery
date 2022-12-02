'''
KNN Classifier:
'''
import numpy as np
class knn:
    #initilize the parameters
    def __init__(self, n_neighbours):
        self.k = n_neighbours
    
    # fit method
    def fit(self, X, y):
        self.X = X
        self.y = y
        return self 
    
    #square euclidean norm 
    def eluclidean_dist(self,a,b):
        return np.dot((a-b),(a-b))
    
    # predict method
    def predict(self, X):
        y_pred = []
        for row in X:
            # compute euclidean distance
            dist = []
            for X_row in self.X:
                dist.append(self.eluclidean_dist(X_row,row))
                
            # get k nearest samples, lables 
            neighbours = np.array(self.y[np.argsort(dist)[:self.k]])
            
            vote = np.bincount(neighbours)                       # voting the closest neighbours
            pred = np.argmax(vote)                                #assign to the mojority 
            y_pred.append(pred)
        return np.array(y_pred)
            