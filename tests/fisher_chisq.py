#!/usr/bin/python3

import sys, json, re
import numpy as np
from scipy.stats import anderson, chi2

# TODO: remove all code for 1000 simulations since we use matrix exponentiation
#       instead of dawg

def main(args):
    p_values = []
    true_p = true_parameters(args)

    for i in range(3,len(args)):
        # get values from json file
        data = open(args[i])
        em_result = json.loads(data.read())

        # calculate chi square
        x2 = chisq(em_result,true_p)
        p_values.append(chi2.cdf(x2,3))

        print(p_values[-1])

def true_parameters(args):
    return get_true_param(args[1])


def chisq(em_result, true_params):
    # extract last values in params
    r = em_result['params'][-1]['r']
    p = em_result['params'][-1]['p']
    q = em_result['params'][-1]['q']
    info = em_result['fisher_info']

    diff = np.array([r,p,q]) - true_params
    return np.matmul(np.matmul(np.transpose(diff),info), diff)

def get_true_param(file):
    f = open(file,'r').readlines()

    # get parameters
    #  aY,aR,b
    m = re.search(r'Subst.Params = [\d., ]+',str(f)).group(0)
    [aY,aR,b] = re.search(r'[\d]+.*',m).group(0).split(',')
    [aY,aR,b] = [float(aY), float(aR), float(b)]
    #  piA,piC,piG,piT
    m = re.search(r'Subst.Freqs = [\d., ]+',str(f)).group(0)
    [piA,piC,piG,piT] = re.search(r'[\d]+.*',m).group(0).split(',')
    [piA,piC,piG,piT] = [float(piA), float(piC), float(piG), float(piT)]
    #  t
    m = re.search(r'Tree.tree = "\(A:[\d.]+',str(f)).group(0)
    t = re.search(r'[\d].+',m).group(0)
    t = float(t)

    # d
    freq = [piA,piC,piG,piT]
    d = 0
    m = np.array([[0,b,aR,b],[b,0,b,aY],[aR,b,0,b],[b,aY,b,0]])
    for i in range(4):
        for j in range(4):
            d += m[i,j]*freq[i]*freq[j]
    # m = np.array([[0,b,aR,b],[b,0,b,aY],[aR,b,0,b],[b,aY,b,0]])
    # m = np.multiply(np.array([piA,piC,piG,piT]),m)
    # d = np.sum(np.sum(np.multiply(np.transpose(m),np.array([piA,piC,piG,piT]))))

    b_em = b/d
    aR_em = (aR/d-b_em)*(piA+piG)
    aY_em = (aY/d-b_em)*(piC+piT)

    # print('aR',aR_em,' aY',aY_em,' b',b_em)
    # print(np.exp(-aR_em*t),np.exp(-aY_em*t),np.exp(-b_em*t))

    # return np.array([np.exp(-aR*t),np.exp(-aY*t),np.exp(-b*t)])
    return np.array([np.exp(-aR_em*t),np.exp(-aY_em*t),np.exp(-b_em*t)])


if __name__ == '__main__':
    main(sys.argv)
