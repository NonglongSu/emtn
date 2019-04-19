#!/usr/bin/python3

import numpy as np

def fisher_E_Step(N,r,p,q,pi):
    # N = N_counts; pi = [piA,piC,piG,piT,piR,piY]

    s_2 = np.zeros((5),dtype=float)

    x = np.zeros((4))
    x[0] = p_type_0([r,p]*2,q,pi[0],pi[4],0,0)    # x_aa
    x[1] = p_type_0([r,p]*2,q,pi[1],pi[5],1,1)    # x_cc
    x[2] = p_type_0([r,p]*2,q,pi[2],pi[4],2,2)    # x_gg
    x[3] = p_type_0([r,p]*2,q,pi[3],pi[5],3,3)    # x_tt

    s_2[0] = exp_2(x[0],N[0][0]) + exp_2(x[2],N[2][2]) + 2*N[0][0]*x[0]*N[2][2]*x[2]
    s_2[1] = exp_2(x[1],N[1][1]) + exp_2(x[3],N[3][3]) + 2*N[1][1]*x[1]*N[3][3]*x[3]

    y = np.zeros((4,4))
    y[0][0] = p_type_1([r,p]*2,q,pi[0],pi[4],0,0)   # y_aa
    y[2][0] = p_type_1([r,p]*2,q,pi[0],pi[4],2,0)   # y_ga
    y[0][2] = p_type_1([r,p]*2,q,pi[2],pi[4],0,2)   # y_ag
    y[2][2] = p_type_1([r,p]*2,q,pi[2],pi[4],2,2)   # y_gg
    y[1][1] = p_type_1([r,p]*2,q,pi[1],pi[5],1,1)   # y_cc
    y[3][1] = p_type_1([r,p]*2,q,pi[1],pi[5],3,1)   # y_tc
    y[1][3] = p_type_1([r,p]*2,q,pi[3],pi[5],1,3)   # y_ct
    y[3][3] = p_type_1([r,p]*2,q,pi[3],pi[5],3,3)   # y_tt

    # E(s3^2) = E(Yaa^2)+E(Yag^2)+E(Yga^2)+E(Ygg^2)+2*Naa*Yaa[Nga*Yga+Nag*Yag+Ngg*Ygg]+
    #        2*Nag*Yag[Nga*Yga+Ngg*Ygg]+2*Nga*Yga*Ngg*Ygg
    # E(s4^2) = E(Ycc^2)+E(Yct^2)+E(Ytc^2)+E(Ytt^2)+2*Ncc*Ycc[Ntc*Ytc+Nct*Yct+Ntt*Ytt]+
    #        2*Nct*Yct[Nga*Ytc+Ntt*Ytt]+2*Ntc*Ytc*Ntt*Ytt

    s_2[2] = exp_2(y[0][0],N[0][0]) + exp_2(y[0][2],N[0][2]) + exp_2(y[2][0],N[2][0]) + \
            exp_2(y[2][2],N[2][2]) + 2*N[0][0]*y[0][0]*(N[0][2]*y[0][2] + N[2][0]*\
            y[2][0] + N[2][2]*y[2][2]) + 2*N[0][2]*y[0][2]*(N[2][0]*y[2][0]+\
            N[2][2]*y[2][2]) + 2*N[2][0]*y[2][0]*N[2][2]*y[2][2]
    s_2[3] = exp_2(y[1][1],N[1][1]) + exp_2(y[1][3],N[1][3]) + exp_2(y[3][1],N[3][1]) + \
            exp_2(y[3][3],N[3][3]) + 2*N[1][1]*y[1][1]*(N[1][3]*y[1][3] + N[3][1]*\
            y[3][1] + N[3][3]*y[3][3]) + 2*N[1][3]*y[1][3]*(N[3][1]*y[3][1]+\
            N[3][3]*y[3][3]) + 2*N[3][1]*y[3][1]*N[3][3]*y[3][3]

    # E(s5^2) = E(Zij^2) + (sum(Zij))^2 - sum(Zij^2)
    # TODO: review E(s5^2)
    z,z2,z_2 = exp_z2(r,p,q,pi,N)
    s_2[4] = z - z2 + z_2

    return(s_2)

# Probability of type 0
def p_type_0(k,q,pi,piK,i,j):
    # k = r if purine or p if pyrimidine
    # piK = piR if purine or piY if pyrimidine
    return (q*k[i])/((q*k[i] if i==j else 0)+q*(1-k[j])*(pi/piK)*\
        (1 if ((i%2)==(j%2)) else 0)+(1-q)*pi)

# Probability of type 1
def p_type_1(k,q,pi,piK,i,j):
    # k = r if purine or p if pyrimidine
    # piK = piR if purine or piY if pyrimidine
    return (q*(1-k[j])*pi/piK)/((q*k[i] if i==j else 0)+q*(1-k[j])*(pi/piK)*\
        (1 if ((i%2)==(j%2)) else 0)+(1-q)*pi)

# Probability of type 2
def p_type_2(k,q,pi,piK,i,j):
    # k = r if purine or p if pyrimidine
    # piK = piR if purine or piY if pyrimidine
    return ((1-q)*pi)/((q*k[i] if i==j else 0)+q*(1-k[j])*(pi/piK)*\
        (1 if ((i%2)==(j%2)) else 0)+(1-q)*pi)

# Expected probability of x^2 or y^2 (e.g. E(Xaa^2), E(Yaa^2))
def exp_2(prob,counts):
    return counts*prob*(1-prob)+(prob*counts)**2

# Expected probability of E(Zij^2)
def exp_z2(r,p,q,pi,N):
    z_2 = 0.0
    z2 = 0.0
    z = np.zeros((4,4),dtype=float)
    k = [r,p]*2
    piK = [pi[0]+pi[2],pi[1]+pi[3]]*2
    for i in range(4):
        for j in range(4):
            z[i,j] = p_type_2(k,q,pi[j],piK[j],i,j)*N[i,j]
            z2 += z[i,j]**2
            z_2 += exp_2(p_type_2(k,q,pi[j],piK[j],i,j),N[i][j])
    return [np.sum(z)**2,z2,z_2]

# B matrix
def B(r,p,q,s):
    b = np.zeros((3,3),dtype=float)
    b[0][0] = s[0]/(r**2) + s[2]/((1-r)**2)
    b[1][1] = s[1]/(p**2) + s[3]/((1-p)**2)
    b[2][2] = (s[0]+s[1]+s[2]+s[3])/(q**2) + s[4]/((1-q)**2)
    return b

# S^2 matrix
def S_2(r,p,q,pi,s,s_2,N,n):
    exp_p = expected_prob(r,p,q,pi,N)
    m = np.zeros((3,3),dtype=float)
    m[0][0] = s_2[0]/(r**2) - 2*(exp_p[0]/((1-r)*r)) + s_2[2]/((1-r)**2)
    m[1][1] = s_2[1]/(p**2) - 2*(exp_p[1]/((1-p)*p)) + s_2[3]/((1-p)**2)
    m[2][2] = s_2[0]/(q**2) + 2*(s[0]*s[1]/(q**2)) + s_2[1]/(q**2)+ \
        2*(exp_p[0]/(q**2))+ 2*(s[1]*s[2]/(q**2)) + s_2[2]/(q**2) + \
        2*((s[0]*s[3]/(q**2))+ exp_p[1]/(q**2) + s[2]*s[3]/(q**2) ) \
        + s_2[3]/(q**2) - (2/((1-q)*q))*( exp_p[2] +exp_p[3] \
        + exp_p[4] + exp_p[5]) + s_2[4]/((1-q)**2)

    # since S*Tranpose[S] is symmetric
    m[0][1] = m[1][0] = s[0]*s[1]/(p*r) - s[1]*s[2]/((1-r)*p) - \
        s[0]*s[3]/((1-p)*r) + s[2]/(1-p) * s[3]/(1-r)

    m[0][2] = m[2][0] = s_2[0]/(q*r) + (s[0]*s[1])/(q*r) - (exp_p[0])/(q*(1-r)) + \
        (exp_p[0])/(q*r) - (s[1]*s[2])/(q*(1-r)) - s_2[2]/(q*(1-r)) + \
        (s[0]*s[3])/(q*r) - (s[2]*s[3])/(q*(1-r)) - exp_p[2]/((1-q)*r) + \
        exp_p[4]/((1-q)*(1-r))

    m[1][2] = m[2][1] = (s[0]*s[1])/(p*q) + s_2[1]/(p*q) + (s[1]*s[2])/(p*q) - \
        (s[0]*s[3])/((1-p)*q) - exp_p[1]/((1-p)*q) + (exp_p[1])/(p*q) - \
        (s[2]*s[3])/((1-p)*q) - s_2[3]/((1-p)*q) - exp_p[3]/(p*(1-q)) + exp_p[5]/((1-p)*(1-q))

    return m

# create individual expected probabilities for each type (0,1,2)
def expected_prob(r,p,q,pi,N):
    m = np.zeros((3,4,4))
    alpha = [r,p]*2
    piK = [pi[0]+pi[2],pi[1]+pi[3]]*2
    for i in range(4):
        m[0][i][i] = p_type_0(alpha,q,pi[i],piK[i],i,i) * N[i][i]
        for j in range(4):
            m[1][i][j] = p_type_1(alpha,q,pi[j],piK[j],i,j) * N[i][j]
            m[2][i][j] = p_type_2(alpha,q,pi[j],piK[j],i,j) * N[i][j]

    # calculate E(s1*s3), E(s2*s4), E(si*s5)
    n = np.zeros((6))
    # E(s1*s3)
    n[0] = exp_a1a2(m[0,0,0],m[1,0,0],N[0,0]) + exp_a1a2(m[0,2,2],m[1,2,2],N[2,2])+\
        m[0,0,0]*(m[1,0,2]+m[1,2,0]+m[1,2,2]) + m[0,2,2]*(m[1,0,0]+m[1,0,2]+m[1,2,0])
    # E(s2*s4)
    n[1] = exp_a1a2(m[0,1,1],m[1,1,1],N[1,1]) + exp_a1a2(m[0,3,3],m[1,3,3],N[3,3])+\
        m[0,1,1]*(m[1,1,3]+m[1,3,1]+m[1,3,3]) + m[0,3,3]*(m[1,1,1]+m[1,1,3]+m[1,3,1])
    # E(s1*s5)
    n[2] = exp_a1a2(m[0,0,0],m[2,0,0],N[0,0]) + exp_a1a2(m[0,2,2],m[2,2,2],N[2,2])+\
        np.sum(m[2,:,:])*(m[0,0,0]+m[0,2,2]) - m[0,0,0]*m[2,0,0] - m[0,2,2]*m[2,2,2]
    # # E(s2*s5)
    n[3] = exp_a1a2(m[0,1,1],m[2,1,1],N[1,1]) + exp_a1a2(m[0,3,3],m[2,3,3],N[3,3])+\
        np.sum(m[2,:,:])*(m[0,1,1]+m[0,3,3]) - m[0,1,1]*m[2,1,1] - m[0,3,3]*m[2,3,3]
    # # E(s3*s5)
    n[4] = exp_a1a2(m[1,0,0],m[2,0,0],N[0,0]) + exp_a1a2(m[1,0,2],m[2,0,2],N[0,2])+\
        exp_a1a2(m[1,2,0],m[2,2,0],N[2,0]) + exp_a1a2(m[1,2,2],m[2,2,2],N[2,2])+\
        np.sum((m[1,0,0]+m[1,0,2]+m[1,2,0]+m[1,2,2])*m[2,:,:])-\
        m[1,0,0]*m[2,0,0] - m[1,0,2]*m[2,0,2] - m[1,2,0]*m[2,2,0]- m[1,2,2]*m[2,2,2]
    # E(s4*s5)
    n[5] = exp_a1a2(m[1,1,1],m[2,1,1],N[1,1]) + exp_a1a2(m[1,1,3],m[2,1,3],N[1,3])+\
        exp_a1a2(m[1,3,1],m[2,3,1],N[3,1]) + exp_a1a2(m[1,3,3],m[2,3,3],N[3,3])+\
        np.sum((m[1,1,1]+m[1,1,3]+m[1,3,1]+m[1,3,3])*m[2,:,:])-\
        m[2,1,1]*m[1,1,1] - m[2,1,3]*m[1,1,3] - m[2,3,1]*m[1,3,1] - m[2,3,3]*m[1,3,3]

    return n

def exp_a1a2(a1,a2,N):
    return a1*a2 - a1*a2/N

# Fisher information matrix
def fisher_info(r,p,q,pi,s,s_2,N,n):
    print('s^2: ',s_2)
    I_y = B(r,p,q,s) - S_2(r,p,q,pi,s,s_2,N,n)
    print('B ')
    print(B(r,p,q,s))
    print('S^2 ')
    print(S_2(r,p,q,pi,s,s_2,N,n))
    print('Inf matrix')
    print(I_y)
    # print('covariance matrix:')
    # print(np.linalg.inv(I_y))

    #return I_y
