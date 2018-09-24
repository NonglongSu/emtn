#!/bin/python3
# Generate matrix frequencies from to sequences

import sys

def base_count(file1,file2):
    # read sequences from dawg
    input1 = open(file1)
    input2 = open(file2)
    seq1_raw = input1.read()
    seq2_raw = input2.read()
    input1.close()
    input2.close()
    
    # convert into char list
    seq1 = list(seq1_raw)
    seq2 = list(seq2_raw)
    
    # check that length of sequences match
    if(len(seq1) != len(seq2)):
        print("sequences have different length!")
	
    # remove newlines
    seq1 = list(filter(lambda b: b != '\n' and b != ' ', seq1))
    seq2 = list(filter(lambda b: b != '\n' and b != ' ', seq2))
    
    # initialize frequency matrix
    freq = {'AA':0, 'AC':0, 'AG':0, 'AT':0, 'CA':0, 'CC':0, 'CG':0, 'CT':0, 'GA':0, 'GC':0, 'GG':0, 'GT':0, 'TA':0, 'TC':0, 'TG':0, 'TT':0}
    
    # count base transitions from seq1 to seq2
    for i in range(len(seq1)):
        freq[seq1[i]+seq2[i]] += 1
    #return values
    #print(list(freq.values()))
    
    return(list(freq.values()))
