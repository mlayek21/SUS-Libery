import numpy as np

"""kmean cluster algorithm"""
class kmeans:
    
    #initilize the parameters
    def __init__(self, n_clusters, max_iter):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        
        self.centroids = None
    
    def init_centroids(self,X):
        cent = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        return cent
    
    def fit_predict(self,X,centriods):
        #used this function for calculating the nearest centroid of the sample
        self.centroids = centriods
        itter_count = 0
        for i in range(self.max_iter):
            #assign clusters
            cluster_group = self.assign_clusters(X)
            old_centroids = self.centroids
            
            #move the centroids
            self.centroids = self.move_centroids(X, cluster_group)
            itter_count +=1
            #check convergance
            if (old_centroids == self.centroids).all():
                break
                
        return cluster_group, self.centroids, itter_count
    
    #creates clusters, i.e. groups points into their appropriate groupings
    def assign_clusters(self,X):
        cluster_group = []
        distances = []
        
        for row in X:
            for centroids in self.centroids:
                distances.append(np.sqrt(np.dot(row-centroids,row-centroids))) #Euclidean distance 
            min_distance = min(distances)
            index_pos = distances.index(min_distance)
            cluster_group.append(index_pos)
            distances.clear()
        return np.array(cluster_group)
    
    def move_centroids(self, X, cluster_group):
        new_centroids = []
        
        cluster_type = np.unique(cluster_group)
        
        for type in cluster_type:
            new_centroids.append(X[cluster_group == type].mean(axis=0))
        
        return np.array(new_centroids)


class kernel_km:
    #initilize methode 
    def __init__(self,k,var):
        self.n_cluster = k
        self.var = var
    
    #initilize the cluster    
    def init_(self,X):
        listClusterMember = [[]for i in range (self.n_cluster)]
        
        #randomly initilize the input data into K_cluster
        origin = np.matrix([[0,0]])
        repeated_cent = np.repeat (origin, X.shape[0], axis = 0)
        X_matrix = abs (np.subtract(X, repeated_cent))
        euclideanMatrix = np.sqrt(np.square(X_matrix).sum(axis=1))
        
        X_new = np.array(np.concatenate((euclideanMatrix, X), axis = 1))
        X_new = X_new[np.argsort(X_new[:, 0])]
        X_new = np.delete(X_new,0,1)
        div = X.shape[0]/ self.n_cluster
        
        for i in range(0, X.shape[0]):
            listClusterMember[int(np.floor(i/div))].append(X_new[i, :])    
        
        return listClusterMember
    
    #Radial_basis_kernel
    def RbfKernel(self,data1, data2, sigma):
        delta =abs(np.subtract(data1, data2))
        squaredEuclidean = (np.square(delta).sum(axis=1))
        result = np.exp(-(squaredEuclidean)/(2*sigma**2))
        return result

    #third term of the equation
    def thirdTerm(self,memberCluster):
        result = 0
        for i in range(0, memberCluster.shape[0]):
            for j in range(0, memberCluster.shape[0]):
                result = result + self.RbfKernel(memberCluster[i, :], memberCluster[j, :], self.var)
        result = result / (memberCluster.shape[0] ** 2)
        return result
    
    #second term of the equation
    def secondTerm(self,dataI, memberCluster):
        result = 0
        for i in range(0, memberCluster.shape[0]):
            result = result + self.RbfKernel(dataI, memberCluster[i,:], self.var)
        result = -2 * result / memberCluster.shape[0]
        return result
    
    def centroid(self,data,shape):
        centroid = np.ndarray(shape)                       #initilising the centroids    
        for i in range (0, self.n_cluster):
            data_cluster = np.asmatrix(data[i])
            cent_cluster = data_cluster.mean(axis = 0)                      #centroid of each cluster
            centroid = np.concatenate((centroid, cent_cluster), axis=0)   #appending the centroids
        
        return centroid
    
    def fit_predict(self,X):
        data = self.init_(X)         #initilize the cluster
        shape = (0,X.shape[1])
        old_cent = self.centroid(data,shape)
        iter_ = 0
        #looping until convergence
        while (True):
            kernel_matrix = np.ndarray(shape=(X.shape[0], 0))               #initilizing the kernel result for all the clusters
            
            #assign data to cluster whose centroid is the closest one
            for i in range (0, self.n_cluster):
                T3 = self.thirdTerm(np.asmatrix(data[i]))
                matrixT3 = np.repeat(T3, X.shape[0], axis=0); matrixT3 = np.asmatrix(matrixT3)
                matrixT2 = np.ndarray(shape=(0,1))
                
                #repeat for all data
                for j in range (0, X.shape[0]):
                    T2 = self.secondTerm(X[j,:], np.asmatrix(data[i]))
                    matrixT2 = np.concatenate((matrixT2, T2), axis=0)
                matrixT2 = np.asmatrix(matrixT2)
                kernel_cluster = np.add(matrixT2,matrixT3)  #kernel result for each cluster
                kernel_matrix = np.concatenate((kernel_matrix, kernel_cluster), axis=1)  #kernel result for all cluster
                
            cluster_matrix = np.ravel(np.argmin(np.matrix(kernel_matrix), axis=1))
            listClusterMember = [[] for l in range(self.n_cluster)]
            
            #assign data to cluster regarding cluster matrix
            for i in range(0, X.shape[0]):
                listClusterMember[(cluster_matrix[i]).item()].append(X[i,:])
            new_cent = self.centroid(listClusterMember,shape)

            #break when converged
            if (old_cent == new_cent).all():
                print("Model converges after "+str(iter_)+" iteration")
                break
            else:
                old_cent=new_cent
                data = listClusterMember
                iter_+=1
        return data, new_cent            


"""
KM++ smarter initilization of the cendroids for better convergence in k mean algorithm
"""
def km_plusplus(X,k):
    
    euclideanMatrixAllCentroid = np.ndarray(shape=(X.shape[0], 0))  #generating blank list for dist
    allCentroid = np.ndarray(shape=(0,X.shape[1]))                 #generating blank list for appending centroids
    first = X[np.random.choice(X.shape[0], 1, replace=False)]     #randomly select the first centroids
    allCentroid = np.concatenate((allCentroid, first), axis=0)     #store the centroid value 
    repeatedCent = np.repeat(first, X.shape[0], axis=0)           
    deltaMatrix = abs(np.subtract(X, repeatedCent))
    euclideanMatrix = np.sqrt(np.square(deltaMatrix).sum(axis=1)) #euclidean dist
    indexNextCent = (np.argmax(np.matrix(euclideanMatrix)))
    
    #start the loop and find the farthest centorids
    if(k>1):
        for a in range(1,k):
            nextCent = np.matrix(X[(indexNextCent).item(),:])
            allCentroid = np.concatenate((allCentroid, nextCent), axis=0)
            for i in range(0,allCentroid.shape[0]):
                repeatedCent = np.repeat(allCentroid[i,:], X.shape[0], axis=0)  #findout the repeated centroids
                deltaMatrix = abs(np.subtract(X, repeatedCent))                   #normilize
                euclideanMatrix = np.sqrt(np.square(deltaMatrix).sum(axis=1))      #euclidean dist
                euclideanMatrixAllCentroid = \
                    np.concatenate((euclideanMatrixAllCentroid, euclideanMatrix), axis=1)
            euclideanFinal = np.min(np.matrix(euclideanMatrixAllCentroid), axis=1)
            indexNextCent = np.argmax(np.matrix(euclideanFinal))
    result = np.array(allCentroid)
    return result

# elbow loss function for km algo, given k value
def loss(X,k):
    km = kmeans(n_clusters=k, max_iter=100)    #intilizing parameters
    centroids = km_plusplus(X, k)                #initilizing the centroids
    y_means = km.fit_predict(X,centroids)          #fit methode to train the model
    loss=0
    for i in range(k):
        for row in X[y_means[0]==i]:
            loss += np.dot(row-y_means[1][i], row-y_means[1][i]) #equation for loss
    return loss
