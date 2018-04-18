
import numpy as np

def EM(tolerance,repeat):
    #randomly generate coefficients
    #   p = exp(-alfaY*t)
    #   q = exp(-beta*t)
    #   r = exp(-alfaR*t)
    p,q,r = 0.9,0.9,0.9
    piA,piC,piG,piT = [0.25]*4#np.random.dirichlet(np.ones(4))   #random values that all add up to 1
    piR = piA+piG
    piY = piC+piT
    S = TN([piA,piC,piG,piT],p,q,r)
    #print(S,'\n',S.sum())
    #NN = np.random.multinomial(10000,np.reshape(S,16))        #values for matrix of transitions
    NN = [2161, 63, 179, 66, 54, 2168, 62, 169, 182, 53, 2190, 61, 68, 157, 76, 2291]
    #NN = [2245, 53, 180, 68, 57, 2174, 72, 168, 152, 71, 2203, 60, 58, 176, 64, 2199]
    
    printMatrix(NN)
    # N:
    #     A     C     G     T
    # A  N[0]  N[1]  N[2]  N[3] 
    # C  N[4]  N[5]  N[6]  N[7]
    # G  N[8]  N[9]  N[10] N[11]
    # T  N[12] N[13] N[14] N[15]

    tol = 0
    for i in range(repeat):
        p,q,r = 0.9,0.9,0.9
        piA,piC,piG,piT = [0.25]*4#np.random.dirichlet(np.ones(4))   #random values that all add up to 1
        piR = piA+piG
        piY = piC+piT

        iteration = 0
        convergence = np.inf
        logLold = -np.inf
        print('tol= ',tolerance[tol])
        while convergence > tolerance[tol]:
            iteration+=1
            #E-step
            S = np.reshape(TN([piA,piC,piG,piT],p,q,r),16)
            N = NN/S
            s1=q*r*(N[0]*piA+N[10]*piG)
            s2=q*p*(N[5]*piC+N[15]*piT)
            s3=q*(1.0-r)*((piA**2*N[0]+piG**2*N[10])+piA*piG*(N[2]+N[8]))/piR
            s4=q*(1.0-p)*((piC**2*N[5]+piT**2*N[15])+piC*piT*(N[7]+N[13]))/piY
            s5=calcS5(np.reshape(N,(4,4)),[piA,piC,piG,piT],q)
            s6=calcS(0,2,[piA,piC,piG,piT],p,q,r,np.reshape(N,(4,4)))
            s7=calcS(2,0,[piA,piC,piG,piT],p,q,r,np.reshape(N,(4,4)))
            s8=calcS(1,3,[piA,piC,piG,piT],p,q,r,np.reshape(N,(4,4)))
            s9=calcS(3,1,[piA,piC,piG,piT],p,q,r,np.reshape(N,(4,4)))
            s10=s3
            s11=s4
            
            #M-step
            r=s1/(s1+s3)
            p=s2/(s2+s4)
            q=(s1+s2+s3+s4)/(s1+s2+s3+s4+s5)        
            piA = (s6*(-s10+s6+s7))/((s6+s7)*(-s10-s11+s6+s7+s8+s9))
            piC = (s8*(-s11+s8+s9))/((s8+s9)*(-s10-s11+s6+s7+s8+s9))
            piG = (s7*(s10-s6-s7))/((s6+s7)*(s10+s11-s6-s7-s8-s9))
            piT = (1.0-piA-piC-piG)
            piR=piA+piG
            piY=piC+piT
            #
            logLnew=logLikelihood([piA,piC,piG,piT],p,q,r,NN)
            if (logLold > logLnew):
                print('log Likelihood error')
                print(iteration,': ',logLnew,'convergence= ',logLnew-logLold,' r=',r,' p=',p,' q=',q,'\n',\
                      ' s6=',s6,' s7=',s7,' s8=',s8,' s9=',s9,' s10=',s10,' s11=',s11,' piA=',piA,' piC=',piC,' piG=',piG,' piT',piT)
                break
            convergence = np.absolute(logLnew-logLold)

            print(iteration,': ',logLnew,'convergence= ',logLnew-logLold,' r=',r,' p=',p,' q=',q,'\n',\
                      ' s6=',s6,' s7=',s7,' s8=',s8,' s9=',s9,' s10=',s10,' s11=',s11,' piA=',piA,' piC=',piC,' piG=',piG,' piT',piT)
            logLold=logLnew
        tol += 1
        print('-------------------------------------------------------------------')
    
def printMatrix(M):
    print('\n',M[0],M[1],M[2],M[3],'\n',M[4],M[5],M[6],M[7],'\n', \
      M[8],M[9],M[10],M[11],'\n',M[12],M[13],M[14],M[15],'\n')

def TN(piVector,p,q,r):
    v = np.zeros((4,4))
    alfaK = [r,p]*2
    piK = [piVector[0]+piVector[2],piVector[1]+piVector[3]]*2
    for i in range(4):
        for j in range(4):
            v[i][j] = (alfaK[i]*q*(1.0 if (i==j) else 0) \
                       + q*(1.0-alfaK[i])*piVector[j]*(1.0 if ((i%2)==(j%2)) else 0)\
                                     /piK[j] \
                       + (1.0-q)*piVector[j])*piVector[i]
    return v

def logLikelihood(piVector,p,q,r,M):
    return (np.log(np.reshape(TN(piVector,p,q,r),16))*M).sum()

def calcS5(N,piVector,q):
    v = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            v[i][j] = (1.0-q)*piVector[j]*piVector[i]*N[i][j]
    return v.sum()

def calcS(i,j,piVector,p,q,r,N):
    y = np.zeros(2)
    z = np.zeros(8)
    alfaK = [r,p]*2
    piK = [piVector[0]+piVector[2],piVector[1]+piVector[3]]
    #calculate X(i,i)
    x = q*alfaK[i]*N[i][i]*piVector[i]
    #calculate Y(-,i)+Y(i,-)
    y[0] = 2*(q*(1.0-alfaK[i])*piVector[i]**2/piK[i%2]*N[i][i])
    y[1] = (q*(1.0-alfaK[i])*piVector[i]*piVector[j]/piK[i%2])*(N[i][j]+N[j][i])
    #calculate Z(-,i)+Z(i,-)
    for k in range(4): 
        z[k*2]   = (1.0-q)*piVector[i]*piVector[k]*N[i][k]
        z[k*2+1] = (1.0-q)*piVector[k]*piVector[i]*N[k][i]
    return y.sum() + z.sum() + x

e=-12
tolerance = [np.power(10.0,e)]#,np.power(10.0,e+1),np.power(10.0,e+2)
EM(tolerance,1)
