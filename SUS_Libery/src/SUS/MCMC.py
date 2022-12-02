import numpy as np

def metropolis_hasting(x, prop_width,mu,sd,mu_,sd_):
    n_iter = 10000                         #no of itteration = generating the n_iter no of sample data
    X_current = 0                       #initilizing the candidate
    candidate = [1]
    b = 0                             #counting the accepetance prob
    PDF_norm = lambda x,mu,sd: np.array(1/(sd * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sd**2) )) 
    for i in range (n_iter):

        like_current = PDF_norm(X_current,mu,sd)                    #likelihood for the candidate having mean=10 & SD = 5
        prior_current = PDF_norm(np.mean(candidate),mu_,sd_)               #prior for the the candidate having mean=25 & SD = 5
        post_current = like_current * prior_current              #posterior update for the current value of the candidate(target distribution i gaussian)

        
        X_purposed = np.random.normal(X_current,prop_width)                         #sampleing a candidate from a normal distribution
        
        
        like_purposed = PDF_norm(X_purposed,mu,sd)                                                 #updated value for the likelihood
        prior_purposed = PDF_norm((np.mean(candidate)+X_purposed/len(candidate)),mu_,sd_)         #updated value for the prior
        post_purposed = like_purposed * prior_purposed                                           #updated value for the posterior

        r = post_purposed / post_current                       #calculating the probablity of acceptence

        a = np.random.rand()                                  #generating random value from a normal distribution

        if r>a:
            X_current = X_purposed                  # updating the purposed value of xandidate
            b+=1
        else:
            X_current = X_current               #or stick to the previous value of the candidate
#             pass
        
        candidate.append(X_current)
    return candidate,b