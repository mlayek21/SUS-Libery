import numpy as np
import pandas as pd
from .MLP import fit as fit
from .MLP import predict as predict
#mean absolute error
def mae(Y,y):
    n = len(Y)
    error = np.sum(np.absolute(Y-y))/n
    return error

def mse(Y,y):
    n = len(Y)
    error = np.sum(np.square(Y-y))/n
    return error

#function for f1 score
def F1_score(y,y_pred):

    #confusion matrix initilizing to 0
    T_neg=0        #true negtive 
    F_neg=0         #false negtive
    T_pos=0          #true positive
    F_pos=0           #false positive
    
    for i in range (len(y)):
        
        if y[i]==0 and y_pred[i]==0:        #model predict negtive lable corectly 
            T_neg+=1 
            
        elif y[i]==1 and y_pred[i]==0:       #model predict negtive lable incorrectly
            F_neg+=1
            
        elif y[i]==1 and y_pred[i]==1:          #model predict positive lable correctly
            T_pos+=1
            
        elif y[i]==0 and y_pred[i]==1:            #modle predict negtive label incorrectly
            F_pos+=1
    
    Errors = F_neg+F_pos                                        #error by the model
    precision = T_pos/(T_pos+F_pos)                               #precision of the model
    accuracy= (T_neg + T_pos)/(T_neg + F_neg + T_pos + F_pos)       #accuracy of the model
    recall = T_pos/(T_pos+F_neg)                                      #recall of the model
    F1_score = 2*precision*recall/(precision+recall)                     #generating the f1 score
    report =[accuracy]+[precision]+[recall]+[F1_score]+[Errors]            #report the data

    return report    

#function for cross validation 
def cross_validation(df,k_fold):
    
    fold=np.array_split(df,k_fold)         
    train = []                       #traning sets list
    test = []                           #test sets list
    cross_val={'train': train, 'test': test}
    
    for i, testi in enumerate(fold):
        train.append(fold[:i] + fold[i+1:])        #split the data frame into 10 folds
        test.append(testi)
    
    report=[]
    for i in range (k_fold):
        X_test = np.array(test[i].loc[:,[ "variance","skewness", "curtosis", "entropy"]])                     #test sets for features
        y_test = np.array(test[i].loc[:,[ "class"]])                                                            #test sets for label
        
        result = pd.concat([train[i][0],train[i][1],train[i][2],train[i][3]], ignore_index=True, sort=False)
        X_train = np.array(result.loc[:,[ "variance","skewness", "curtosis", "entropy"]])                       #train sets for features
        y_train = np.array(result.loc[:,[ "class"]])                                                           #train sets for label
        
        W = fit(X_train,y_train)                    #train the model from the model and appending the weights
        
        y_pred = predict(X_test,W)                   #predict the label from test sets 
        y_test=y_test.reshape(y_pred.shape)
    
        
        f1 = F1_score(y_test,y_pred)          #generate the f1 score by comparing the model prection and actual label 
        report.append(f1)                          #apending the report
    return report

def confusion_matrix(Y,y,a):
    pred = y   #predicted values
    true = Y    #actual values
    classes = len(np.unique(ytest))
    #computing the confusion matrix
    conf_matrix = np.bincount(true * classes + pred).reshape((classes, classes))
    
    # Print the confusion matrix using Matplotlib
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.5)
    ticks = ['Cat','Dog']
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    ax.set_xticks(np.arange(len(ticks)), labels=ticks, fontsize=12)
    ax.set_yticks(np.arange(len(ticks)), labels=ticks, fontsize=12)
    plt.xlabel('Predictions', fontsize=16)
    plt.ylabel('Actuals', fontsize=16)
    plt.title('Confusion Matrix '+ a, fontsize=22)
    plt.show()
    return np.ravel(conf_matrix)