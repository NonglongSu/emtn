#!/usr/bin/python3

import sys, json, re
import numpy as np
from scipy.stats import anderson

def main(args):
    chi_square = []
    true_params = get_true_param(args[1])

    for i in range(2,len(args)):
        # get values from json file
        data = open(args[i])
        # chisq_fisher.append(json.loads(data.read())['fisher_info'])
        em_result = json.loads(data.read())
        # print(em_result['params'][-1]['r'])

        # calculate chi square
        chi_square.append(chisq(em_result,true_params))

    # anderson-darling test
    ad_test = anderson(chi_square,dist='norm')
    print('Critical value at alpha = 5% for',len(args)-2,'runs is:'\
        ,ad_test[1][2])


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
    m = np.array([[0,b,aR,b],[b,0,b,aY],[aR,b,0,b],[b,aY,b,0]])
    m = np.array([piA,piC,piG,piT])*m
    d = sum(np.matmul(np.transpose(m),np.array([piA,piC,piG,piT])))

    b_em = b/d
    aR_em = (aR/d-b_em)*(piA+piG)
    aY_em = (aY/d-b_em)*(piC+piT)

    return np.array([np.exp(-aR_em*t),np.exp(-aY_em*t),np.exp(-b_em*t)])


if __name__ == '__main__':
    main(sys.argv)
