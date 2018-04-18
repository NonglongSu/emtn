
import numpy as np

## p = exp(-alpha*t)
## q = exp(-beta*t)

# p = exp(-t*(2*R+1)/(R+1))
# q = exp(-(2*t)/(R+1))

def EM(tolerance):
    #randomly generated coefficients
    ##R = alpha / (2.0*beta)      #transitions/transversions ratio
    R = 2.0
    p,q = np.random.rand(2)
    alpha = np.random.rand(1)
    beta = (1.0-alpha)/2.0      #alpha + 2*beta = 1
    print('p: ',p,' q: ',q,' alpha: ',alpha,' beta: ',beta,' R:',R,'\n')
    S = K2P(p,q)
    printMatrix(np.reshape(S,16));
    print('sum: ',S.sum(),'\n')
    NN = np.random.multinomial(10000,np.reshape(S,16))
    printMatrix(NN)

    iteration = 0
    convergence = np.inf
    logLold = -np.inf

    while convergence > tolerance:
        iteration += 1

        #E-step
        S = np. reshape

def printMatrix(M):
    print('\n',M[0],M[1],M[2],M[3],'\n',M[4],M[5],M[6],M[7],'\n', \
      M[8],M[9],M[10],M[11],'\n',M[12],M[13],M[14],M[15],'\n')

def K2P(p,q):
    v = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
           v[i][j] = (p*q*(1.0 if (i==j) else 0) \
                     + q*(1.0-p)*0.25*(1.0 if ((i%2)==(j%2)) else 0) \
                        /0.5
                     + (1-q)*0.25)*0.25
    return v

def logLikelihood(alpha,beta,M):
    return (np.log(K2P(alpha,beta))*M).sum()

def calcParameters(alpha,beta):
    R = alpha/(2.0*beta)

tol = np.power(10.0,-12)
EM(tol)


