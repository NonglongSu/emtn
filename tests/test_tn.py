#!/bin/python3

# Tests for Tamura Nei model

import os, sys, numpy as np
sys.path.append("../")
import TN

def create_dawg():
    f = open('../Data/tn.dawg','w')
    # remember dawg order is aY,aR,b
    str_out = ['Tree.tree = "(A:0.2)B;"\n', 'Subst.Model = "TN"\n', 'Subst.Params = 0.5, 0.4, 0.2\n', 'Subst.Freqs = 0.3,0.2,0.2,0.3\n', 'Root.Length = 1000000\n', 'Sim.reps = 3\n']
    f.writelines(str_out)
    f.close()

def run_dawg():
    os.system("make -s -C ../Data/ tn.1.fasta tn.2.fasta")

def run_EM():
    return TN.EM_TN(np.power(10.0,-15),TN.readFreqMatrix("../Data/tn.1.fasta","../Data/tn.2.fasta"))

def get_rates(estimates):
    # convert estimates to dawg values
    #   d is the normalization value
    d = 0.208
    aR = (estimates[1]/(estimates[4]+estimates[6])+estimates[3])*d
    aY = (estimates[2]/(estimates[5]+estimates[7])+estimates[3])*d
    b = estimates[3]*d
    return [aR,aY,b]

def check_range(value,estimate):
    # epsilon is 1%
    return (abs(value-estimate)/value < 0.01)

def test_EM_TN(estimates):
    # create true values array
    #          [t,  aR, aY, b,  piA,piC,piG,piT]
    t_values = [0.2,0.4,0.5,0.2,0.3,0.2,0.2,0.3]
    values = ['t','aR','aY','b','piA','piC','piG','piT']
    # convert estimates to dawg values
    estimates[1:4] = get_rates(estimates)

    e = 0
    # check parameters
    for i in range(8):
        #assert(check_range(estimates[i],t_values[i]) == True)
        if(check_range(estimates[i],t_values[i]) == False):
            print('Assertion failed.... ',values[i],': ',estimates[i],' != ',t_values[i])
            e += 1
        else:
            print(values[i],': ',estimates[i])
    return e

def main():
    create_dawg()
    run_dawg()
    est_params = run_EM()
    if(type(est_params) is not list):
        #log-likelihood error ocurred
        print('Test failed: log-likelihood error')
    else:
        result = test_EM_TN(est_params)
        if(result == 0):
            print('All tests passed!')
        else:
            print(result,' tests failed.')

if __name__ == '__main__':
    main()
