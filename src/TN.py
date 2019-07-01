#!/usr/bin/python3

import numpy as np
import sys, json
from scipy.linalg import expm, inv
from scipy.stats import chi2
import lib, fisher

def EM_TN(tolerance,json_out,no_print,N_counts = 'multinomial',test = False):
    #   p = exp(-alfaY*t)
    #   q = exp(-beta*t)
    #   r = exp(-alfaR*t)

    # N:
    #       A        C        G        T
    # A  N[0][0]  N[0][1]  N[0][2]  N[0][3]
    # C  N[1][0]  N[1][1]  N[1][2]  N[1][3]
    # G  N[2][0]  N[2][1]  N[2][2]  N[2][3]
    # T  N[3][0]  N[3][1]  N[3][2]  N[3][3]

    if(N_counts == 'multinomial'):
        # Create counts matrix from a random distribution
        temp = np.random.dirichlet(np.ones(2))
        while temp[0]<0.1 or temp[1]<0.1:
            temp = np.random.dirichlet(np.ones(2))
        freq = np.array([temp[0]/2,temp[1]/2,temp[1]/2,temp[0]/2])

        fR = freq[0]+freq[2]
        fY = freq[1]+freq[3]

        t = np.random.uniform(low=0.1,high=0.3)
        rho = np.random.normal(1,0.1)
        # rho = np.random.uniform(low=0.5,high=1.5)
        R = np.random.uniform(low=max((freq[0]*freq[2]+freq[1]*freq[3])/\
            (fR*fY),rho)+0.2, high=5.0)

        b = 1/(2*fR*fY*(1+R))
        aY = (fR*fY*R-freq[0]*freq[2]-freq[1]*freq[3])/(2*(1+R)*\
            (fY*freq[0]*freq[2]*rho+fR*freq[1]*freq[3]))
        aR = rho*aY

        r,p,q = np.exp(-t*aR),np.exp(-t*aY),np.exp(-t*b)
        if(not test):
            print('true t,r,p,q',t,r,p,q)
        true_params = [r,p,q]

        S = lib.TN(freq,p,q,r)
        N_counts = np.random.multinomial(1000000,np.reshape(S,16))
        N_counts = np.reshape(N_counts,(4,4))
    else:
        N_counts = lib.readFreqMatrix(N_counts)

    piA = lib.initialPi(0,N_counts)
    piC = lib.initialPi(1,N_counts)
    piG = lib.initialPi(2,N_counts)
    piT = lib.initialPi(3,N_counts)
    piR = piA+piG
    piY = piC+piT

    # estimate initial values for p,q,r,t using TN distance formula
    p,q,r,t = lib.initialParameters(piA,piC,piG,piT,N_counts)
    params = np.array([p,q,r])

    iteration = 0
    max_iterations = 1000
    p_val = np.inf
    info = np.ones((3,3),dtype=float)
    logLold = -np.inf
    logLnew = 1
    log_diff = np.inf

    json_data = {}
    json_data['params'] = []

    # while (p_val > tolerance and iteration < max_iterations) or iteration < 10 :
    while (log_diff > tolerance and iteration < max_iterations) or iteration < 10 :

        iteration += 1

        #E-step
        S = lib.TN([piA,piC,piG,piT],p,q,r)
        N = N_counts/S
        s1=q*r*(N[0][0]*piA+N[2][2]*piG)
        s2=q*p*(N[1][1]*piC+N[3][3]*piT)
        s3=q*(1.0-r)/piR*((piA**2.0*N[0][0]+piG**2.0*N[2][2])+piA*piG*(N[0][2]+N[2][0]))
        s4=q*(1.0-p)/piY*((piC**2.0*N[1][1]+piT**2.0*N[3][3])+piC*piT*(N[1][3]+N[3][1]))
        s5=lib.calcS5(N,[piA,piC,piG,piT],q)
        s6=lib.calcS(0,2,[piA,piC,piG,piT],p,q,r,N)
        s7=lib.calcS(2,0,[piA,piC,piG,piT],p,q,r,N)
        s8=lib.calcS(1,3,[piA,piC,piG,piT],p,q,r,N)
        s9=lib.calcS(3,1,[piA,piC,piG,piT],p,q,r,N)
        s10=s3
        s11=s4

        # Fisher E-step
        s_2 = fisher.fisher_E_Step(N_counts,r,p,q,[piA,piC,piG,piT,piR,piY])

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

        #Calculation of R,t,rho
        R,t,rho = lib.calcParameters(piA,piC,piG,piT,p,q,r)
        logLnew = lib.logLikelihood([piA,piC,piG,piT],p,q,r,N_counts)

        # if (logLold > logLnew):
        #     print('LOG LIKELIHOOD ERROR. Iteration ',iteration)
        #     print('%d r: %.8f p: %.8f q: %.8f LogLhood: %.10f convergence: %.5e ' % \
        #         (iteration,r,p,q,logLnew,chisq))
        #     print('loglikelihood diff: ',logLnew-logLold)
        #     return 1

        theta_diff = np.array([p,q,r]) - params
        chisq = np.matmul(np.matmul(np.transpose(theta_diff),\
            info), theta_diff)
        p_val = chi2.cdf(chisq,3)

        info = fisher.fisher_info(r,p,q,[piA,piC,piG,piT],[s1,s2,s3,s4,s5],\
            s_2,N_counts)

        log_diff = logLnew-logLold
        logLold=logLnew
        params = np.array([p,q,r])

        if(not no_print):
            print('%d t:%.8f r:%.8f p:%.8f q:%.8f LogLhood:%.10f p-val:%.5e ' % \
                (iteration,t,r,p,q,logLnew,p_val))

        json_data['params'].append({"r":r, "p":p, "q":q, "LogLhood":logLnew, "p_value":p_val})
        # end of while loop

    fisher_information = fisher.fisher_info(r,p,q,[piA,piC,piG,piT],[s1,s2,s3,s4,s5],s_2,N_counts)
    if(not no_print):
        print('\nfisher information:\n',fisher_information)

    # append info to output json
    json_data['chisq'] = chisq
    json_data['fisher_info'] = fisher_information.tolist()
    json_data['iterations'] = iteration

    # save data in json file
    with open(json_out,'w') as outfile:
        json.dump(json_data,outfile)

    if(test):
        diff = np.array([r,p,q]) - true_params
        x2_validation = np.matmul(np.matmul(np.transpose(diff),fisher_information), diff)
        p_val = chi2.cdf(np.matmul(np.matmul(np.transpose(diff),fisher_information), diff),3)

        with open('em_chi_values.csv','a+') as outputfile:
            outputfile.write(str(p_val)+","+str(R)+","+str(rho)+","+str(freq)+"\n")

        with open('iterations.csv','a+') as outit:
            outit.write(str(iteration)+"\n")
