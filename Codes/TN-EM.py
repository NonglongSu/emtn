
import numpy as np

def EM(tolerance):
    #randomly generate coefficients
    #   p = exp(-alfaY*t)
    #   q = exp(-beta*t)
    #   r = exp(-alfaR*t)
    p,q,r = 0.9,0.9,0.9
    piA,piC,piG,piT = [0.25]*4#np.random.dirichlet(np.ones(4))   #random values that all add up to 1
    piR = piA+piG
    piY = piC+piT
    S = TN([piA,piC,piG,piT],p,q,r)
    print(S,'\n',S.sum())
    NN = np.random.multinomial(20000,S.reshape(16))        #values for matrix of transitions
    printMatrix(NN)
    # N:
    #     A     C     G     T
    # A  N[0]  N[1]  N[2]  N[3] 
    # C  N[4]  N[5]  N[6]  N[7]
    # G  N[8]  N[9]  N[10] N[11]
    # T  N[12] N[13] N[14] N[15]

    #printMatrix(N)
    convergence = np.inf
    logLold = -np.inf
    while convergence > tolerance:
        #E-step
        S = TN([piA,piC,piG,piT],p,q,r)
        N = NN/np.reshape(S,16)
        
        s1=q*r*(N[0]*piA+N[10]*piG)
        s2=q*r*(N[5]*piC+N[15]*piT)
        s3=q*(1-r)*(piA**2/piR*N[0]+piG**2/piR*N[10]+piA*piG/piR*(N[2]+N[8]))
        s4=q*(1-p)*(piC**2/piY*N[5]+piT**2/piY*N[15]+piC*piT/piY*(N[7]+N[13]))
        s5=NN.sum()-(s1+s2+s3+s4)
        s6=N[0]+N[1]+N[2]+N[3]+N[4]+N[8]+N[12]+N[0]-(q*r*N[0]*piA)
        s7=N[10]+N[8]+N[9]+N[11]+N[2]+N[6]+N[14]+N[10]-(q*r*N[10]*piG)
        s8=N[5]+N[4]+N[6]+N[7]+N[1]+N[9]+N[13]+N[5]-(q*p*N[5]*piC)
        s9=N[15]+N[12]+N[13]+N[14]+N[3]+N[7]+N[11]+N[15]-(q*p*N[15]*piT)
        s10=s3
        s11=s4
        #M-step
        r=s1/(s1+s3)
        p=s2/(s2+s4)
        #q=(s1+s2+s3+s4)/(s1+s2+s3+s4+s5)
        #piA = (-piT*s10*s6+piT*s6**2+piT*s6*s7)/((s6+s7)*s9)
        #piC = (piT*s8*(piT*s10-piT*s6-piT*s7+s9))/(s9*(piT*s10+piT*s11-piT*s6-piT*s7+s9))
        #piG = (-piT*s10*s7+piT*s6*s7+piT*s7**2)/((s6+7)*s9)
        #piT = (1-piA-piC-piG if (1-piA-piC-piG)>0 else np.finfo(float).eps) #prevent piT to be a negative number
        #piR=piA+piG
        #piY=piC+piT
        #
        logLnew=logLikelihood([piA,piC,piG,piT],p,q,r,NN)
        if (logLold > logLnew):
            print('log Likelihood error')
            print(logLnew,' r=',r,' p=',p,' q=',q,' s1=',s1,' s2=',s2,' s3=',s3,' s4=',s4,', s5=',s5)
            break
        convergence = np.absolute(logLold-logLnew)
        logLold=logLnew
        print(logLnew,' r=',r,' p=',p,' q=',q,' s1=',s1,' s2=',s2,' s3=',s3,' s4=',s4,', s5=',s5)
              #\,' piA=',piA,' piC=',piC,' piG=',piG,' piT',piT)

def printMatrix(M):
    print('\n',M[0],M[1],M[2],M[3],'\n',M[4],M[5],M[6],M[7],'\n', \
      M[8],M[9],M[10],M[11],'\n',M[12],M[13],M[14],M[15],'\n')

def TN(piVector,p,q,r):
    v = np.zeros((4,4))
    alfaK = [r,p]*2
    piK = [piVector[0]+piVector[2],piVector[1]+piVector[3]]*2
    for i in range(4):
        for j in range(4):
            v[i][j] = (alfaK[i]*q*(1 if (i==j) else 0) \
                       + q*(1-alfaK[i])*piVector[j]*(1 if ((i%2)==(j%2)) else 0)\
                                     /piK[j] \
                       + (1-q)*piVector[j])*piVector[i]
    return v

def logLikelihood(piVector,p,q,r,M):
    return (np.log(np.reshape(TN(piVector,p,q,r),16))*M).sum()

tolerance = np.power(10.,-3)    #by increasing tolerance to 10^(-3), instead of 10^(-6) we encounter less logL errors
EM(tolerance)
