#!/usr/bin/python3

import numpy as np

def fisher_E_Step(N,r,p,q,pi):
    # N = N_counts; pi = [piA,piC,piG,piT,piR,piY]

    x[0] = p_type_0(r,q,pi[0],pi[4])    # x_aa
    x[1] = p_type_0(r,q,pi[1],pi[5])    # x_cc
    x[2] = p_type_0(p,q,pi[2],pi[4])    # x_gg
    x[3] = p_type_0(p,q,pi[3],pi[5])    # x_tt

    s1_2 = exp_z2(x[0],N[0][0]) + exp_z2(x[2],N[2][2]) + 2 * x[0] * x[2]
    s2_2 = exp_z2(x[1],N[1][1]) + exp_z2(x[3],N[3][3]) + 2 * x[1] * x[3]

    y[0][0] = y[2][0] = p_type_1(r,q,pi[0],pi[4])   # y_aa , y_ga
    y[0][2] = y[2][2] = p_type_1(r,q,pi[2],pi[4])   # y_ag , y_gg
    y[1][1] = y[3][1] = p_type_1(p,q,pi[1],pi[5])   # y_cc , y_tc
    y[1][3] = y[3][3] = p_type_1(p,q,pi[3],pi[5])   # y_ct , y_tt

    # TODO: think about making a for loop
    s3_2 = exp_z2(y[0][0],N[0][0]) +
    s4_2 =
#    s5_2 =

# Probability of type 0
def p_type_0(k,q,pi,piK):
    # k = r if purine or p if pyrimidine
    # piK = piR if purine or piY if pyrimidine
    return (q*k)/(q*k+q*(1-k)*(pi/piK)+(1-q)*pi)

def p_type_1(k,q,pi,piK)
    # k = r if purine or p if pyrimidine
    # piK = piR if purine or piY if pyrimidine
    return (q*(1-k))/(q*r+q*(1-r)*(pi/piK)+(1-q)*pi)

def exp_z2(prob,counts):
    # calculate expected probility of z^2 (e.g. E(Xaa^2))
    return counts*prob*(1-prob)+(prob*counts)**2
