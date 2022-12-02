import numpy as np
class SVC:
    def __init__(self, kernal='Linear',learning_rate = 1e-1, ntol=1e-5,maxiter=5000, Regularizar=0.1 ):
        self.kernal = kernal
        self.l = learning_rate
        self.tol = ntol
        self.iter = maxiter
        self.lamb = Regularizar
        self.y = None
        self.w = None
        
    def kernal(self):
        pass
    
    def loss(self,a,b):
        L = 1-(a@self.w.T)*b
        # L[L <= 0] = 0
        loss = np.float64(self.w@self.w.T)+1/len(L)*np.sum(L)
        return loss
    
    def dloss(self,a,b):
        L = (a@self.w.T)*b
        L[L >= 1] = 0
        ind = np.where(L.T[0]<0)
        return b[ind][:].T@a[ind][:]
         
    def fit(self,X,y):
        self.x = X
        self.y = y
        self.x = np.hstack([np.ones((self.x.shape[0],1)),self.x])
        self.w = np.ones((1,self.x.shape[1]))
        self.y[self.y == 0] = -1
        self.y = (self.y).reshape(self.y.shape[0],1)
        loss=[]
        temp = []
        i = 0
        while True:
            derror = 2*self.lamb*self.w - self.dloss(self.x,self.y)
            temp.append(self.w)
            self.w-= self.l*derror
            loss.append(self.loss(self.x,self.y))
            i+=1
            if i<=50:
                continue
            else:
                if i>=self.iter or np.linalg.norm(np.average(temp[-50])-self.w)<=self.tol:
                    self.y[self.y == -1] = 0
                    print('Model converges: '+str(not(i==self.iter))+' '+str(i)+' iteration')
                    return loss
                
    def predict(self,x):
        x = np.hstack([np.ones((x.shape[0],1)),x])
        ypred =  np.sign(x@self.w.T)
        ypred[ypred<0]=0
        return ypred.astype('int32').ravel()