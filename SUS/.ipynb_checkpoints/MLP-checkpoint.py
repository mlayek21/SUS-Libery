import numpy as np
#stocastically picking a random feature and label for the traning set
def rnd (x,y):
    n = np.random.randint(x.shape[0])   
    a=x[n]
    b=y[n]

    if y[n]==0:                    #converting the 0 label to -1
        b=-1
    else:
        b=1
        pass
    return a,b

def fit(X,y):
    w = np.array(np.ones(X.shape[1]))              #initilizing the weights to 1
    lr = 1                                        #seting learning rate to 1
    a = 1                                        #count for model make mistake 
    b = 1                                       #count for model does not mistake
    
    while True:
        
        m=rnd(X,y)                      #picking random values for our feature and label from the test set
        x_ = m[0]
        y_ = m[1]
        z = y_*w.dot(x_)           #perceptron loss function

        if z<0:                    #conditional statement model update weights for negtive loss
            w += (lr*x_*y_)
            z = y_*w.dot(x_)
            a+=1
            
        else:                       #otherwise model does not update weights
            b+=1
            pass

        if b == 10*a:         # Weights are optimise well enough that, model make 10 time less error 
            break

    return w

#predict function
def predict (X,W):
    
    y_pred = []                       
    
    for i in range (X.shape[0]):
        z = W.T.dot(X[i])         #loss function
        
        if z >= 0:          #conditional statement if our loss is greater thean equeal to 0 i.e model predict label =1
            a=1
        else :              #otherwise for negtive loss lable is 0 
            a=0
    
        y_pred.append(a)          #apending our model prediction
    return np.asarray(y_pred)