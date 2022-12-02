from scipy.stats import multivariate_normal   #importing extra libraries
import numpy as np

def log_sum_exp(Z):
    """ Compute log(\sum_i exp(Z_i)) for some array Z."""
    return np.max(Z) + np.log(np.sum(np.exp(Z - np.max(Z))))

def loglikelihood(data, weights, means, covs):
    """ Compute the loglikelihood of the data for a Gaussian mixture model with the given parameters. """
    num_clusters = len(means)
    num_dim = len(data[0])
    
    ll = 0
    for d in data:
        
        Z = np.zeros(num_clusters)
        for k in range(num_clusters):

            delta = np.array(d) - means[k]
            exponent_term = np.dot(delta.T, np.dot(np.linalg.inv(covs[k]), delta))
            
            # Compute loglikelihood contribution for this data point and this cluster
            Z[k] += np.log(weights[k])
            Z[k] -= 1/2. * (num_dim * np.log(2*np.pi) + np.log(np.linalg.det(covs[k])) + exponent_term)
            
        # Increment loglikelihood contribution of this data point across all clusters
        ll += log_sum_exp(Z)
        
    return ll

def compute_responsibilities(data, weights, means, covariances):
    '''E-step: compute responsibilities, given the current parameters'''
    num_data = len(data)
    num_clusters = len(means)
    resp = np.zeros((num_data, num_clusters))
    
    # Update resp matrix so that resp[i,k] is the responsibility of cluster k for data point i.
    for i in range(num_data):
        for k in range(num_clusters):
            resp[i, k] = weights[k]*multivariate_normal.pdf(data[i], mean=means[k], cov=covariances[k])
    
    # Add up responsibilities over each data point and normalize
    row_sums = resp.sum(axis=1)[:, np.newaxis]
    resp = resp / row_sums
    
    return resp
    '''M steps Updating the parameters'''
def compute_soft_counts(resp):
    # Compute the total responsibility assigned to each cluster, which will be useful when 
    counts = np.sum(resp, axis=0)
    return counts

def compute_weights(counts):
    num_clusters = len(counts)
    weights = [0.] * num_clusters
    
    for k in range(num_clusters):
        # Update the weight for cluster k using the M-step update rule for the cluster weight, \hat{\pi}_k.

        weights[k] = counts[k] / np.sum(counts)

    return weights

def compute_means(data, resp, counts):
    num_clusters = len(counts)
    num_data = len(data)
    means = [np.zeros(len(data[0]))] * num_clusters
    
    for k in range(num_clusters):
        # Update means for cluster k using the M-step update rule for the mean variables.
        weighted_sum = 0.
        for i in range(num_data):

            weighted_sum += data[i] * resp[i][k]

        means[k] = weighted_sum / counts[k]

    return means

def compute_covariances(data, resp, counts, means):
    num_clusters = len(counts)
    num_dim = len(data[0])
    num_data = len(data)
    covariances = [np.zeros((num_dim,num_dim))] * num_clusters
    
    for k in range(num_clusters):
        # Update covariances for cluster k using the M-step update rule for covariance variables.

        weighted_sum = np.zeros((num_dim, num_dim))
        for i in range(num_data):
            weighted_sum += resp[i][k]*np.outer(data[i] - means[k], data[i] - means[k])

        covariances[k] = weighted_sum / counts[k]

    return covariances

#assign the cluster
def assign (X,z):
    cluster1 = []   #cluster1
    cluster2 = []   #cluster2
    cluster3 = []   #cluster3
    for i in range (len(X)):
        c = np.where(z[i]==np.max(z[i]))[0][0]
        if c==0:
            cluster1.append(X[i])
        elif c==1:
            cluster2.append(X[i])

        elif c==2:
            cluster3.append(X[i]) 
    return np.vstack(cluster1), np.vstack(cluster2), np.vstack(cluster3)


def EM(data, init_means, init_covariances, init_weights, maxiter=1000, thresh=0.1):
    
    # Make copies of initial parameters, which we will update during each iteration
    means = init_means[:]
    covariances = init_covariances[:]
    weights = init_weights[:]
    
    # Infer dimensions of dataset and the number of clusters
    num_data = len(data)
    num_dim = len(data[0])
    num_clusters = len(means)
    
    # Initialize some useful variables
    resp = np.zeros((num_data, num_clusters))
    ll = loglikelihood(data, weights, means, covariances)
    ll_trace = [ll]
    itter_ = 0
    
    for it in range(maxiter):

        # E-step: compute responsibilities
        resp = compute_responsibilities(data, weights, means, covariances)

        # M-step
        counts = compute_soft_counts(resp)
        
        # Update the weight for cluster k using the M-step update rule for the cluster weight, \hat{\pi}_k.
        weights = compute_weights(counts)
        
        # Update means for cluster k using the M-step update rule for the mean variables.
        means = compute_means(data, resp, counts)
        
        # Update covariances for cluster k using the M-step update rule for covariance variables.
        covariances = compute_covariances(data, resp, counts, means)
        
        # Compute the loglikelihood at this iteration
        ll_latest = loglikelihood(data, weights, means, covariances)
        ll_trace.append(ll_latest)
        itter_+=1
        # Check for convergence in log-likelihood and store
        if (ll_latest - ll) < thresh and ll_latest > -np.inf:
            print("Model converges after "+str(itter_)+" iteration")
            break
        ll = ll_latest
    
    clusters = assign(data, resp)  #assigning the clusters
    return clusters, means
