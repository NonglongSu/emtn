import numpy as np

def init():
    # mean and standard deviation for sample data
    s1 = np.random.normal(0,1, (200,1))
    s2 = np.random.normal(9,1, (100,1))

    x = np.vstack((s1,s2))
    np.random.shuffle(x)
    
    return x

def EM(x,eps=0.00001,max_iter=1000):
    # obtain number of data points (n) and dimension (d)
    n, d = x.shape
    
    #randomly choose starting means
    mu = x[np.random.choice(n,2,False),:]

    #initialize covariance matrices
    sigma = [np.eye(d)] * 2

    #initialize probabilities for each gaussians
    prob = [1/2] * 2
    
    #initialize responsibility matrix
    R = np.zeros((n,2))

    log_likelihood = []

    #probability function definition
    P = lambda mu, s: np.linalg.det(s) ** -.5 * (2 * np.pi) ** (-d/2.) \
                * np.exp(-.5 * np.einsum('ij, ij -> i',\
                x - mu, np.dot(np.linalg.inv(s) , (x - mu).T).T ) )

    convergence = False

    while ((len(log_likelihood) < max_iter) and (not convergence)):
        ## E-step ##

        #calculation of the membership of each of the 2 gaussians
        for k in range(2):
            R[:,k] = prob[k] * P(mu[k], sigma[k])

        #likelihood computation
        log_likelihood.append(np.sum(np.log(np.sum(R,axis=1))))

        # Normalize so that the responsibility matrix is row stochastic
        R = (R.T / np.sum(R, axis = 1)).T

        #Number of data points from each gaussian
        N = np.sum(R,axis=0)

        ## M-step ##

        #calculate new mean and covarianze for each gaussian
        for k in range(2):

            #mean
            mu[k] = 1. / N[k] * np.sum(R[:,k] * x.T, axis=1)
            x_mu = np.matrix(x - mu[k])

            #covariance
            sigma[k] = np.array( 1. / N[k] * np.dot(np.multiply(x_mu.T, R[:,k]), x_mu))

            #probabilities
            prob[k] = 1. / n * N[k]

        try:
            convergence = np.abs(log_likelihood[-1]-log_likelihood[-2])<eps
        except IndexError:
            continue
    print ('prob: ',prob,'\n','iterations: ',len(log_likelihood))

x = init()
EM(x)
