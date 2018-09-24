
import numpy as np

def EM(tolerance):
    #randomly generate coefficients
    #   p = exp(-alfa_y*t)
    #   q = exp(-beta*t)
    #   r = exp(-alfa_r*t)
    p,q,r = np.random.rand(3)                           #random values from interval [0,1)
    piA,piC,piG,piT = np.random.dirichlet(np.ones(4))   #random values that all add up to 1
    piR = piA+piG
    N = np.random.multinomial(200,[1/16.]*16)           #random values for matrix of transitions
    # N:                                                 assuming all transitions have same ratio
    #     A     C     G     T
    # A  N[0]  N[1]  N[2]  N[3] 
    # C  N[4]  N[5]  N[6]  N[7]
    # G  N[8]  N[9]  N[10] N[11]
    # T  N[12] N[13] N[14] N[15]

    #printMatrix(N)
    convergence = np.inf
    while convergence > tolerance:
        #E-step
        s1=q*r*(N[0]*piA+N[10]*piG)*(np.log(r)+np.log(q))
        s3=q*(1-r)*(piA**2/piR*N[0]+piG**2/piR*N[10]+piA*piG/piR*(N[2]+N[8]))\
            *(np.log(q)+np.log(1-r))
        convergence = np.absolute(r-(s1/(s1+s3)))
        #M-step
        r=s1/(s1+s3)
    print('s1: ',s1,', s3: ',s3)

def printMatrix(M):
    print(M[0],M[1],M[2],M[3],'\n',M[4],M[5],M[6],M[7],'\n', \
      M[8],M[9],M[10],M[11],'\n',M[12],M[13],M[14],M[15])

#def fullLikelihood():

def generateTNsamples(piA,piC,piG,piT,p,q,r):
    piR = piA + piG
    piY = piC + piT
    #log(q) = -beta*t
    #log(r) = -alfaR*t
    #log(p) = -alfaY*t
    [piA, alfaR*piG/piR+beta*piG, beta*piC, beta*piT, \
     alfaR*piA/piR+beta*piA, piG, beta*piC, beta*piT, \
     beta*piA, beta*piG, piC, alfaY*betaT/piY+beta*piT, \
     beta*piA, beta*piG, alfaY*piC/piY+beta*piC, piT]
    

tolerance = np.power(10.,-6)
EM(tolerance)
