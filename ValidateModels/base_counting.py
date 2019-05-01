#!/usr/bin/python3

# Generate matrix counts from sequences

import sys

def base_count(file1): #,file2):
    # read sequences from dawg
    fasta = open(file1).readlines()
    seq1_raw = fasta[1:fasta.index('\n')]
    seq2_raw = fasta[fasta.index('\n')+2:]
    seq1 = list(filter(lambda char: char != '\n',''.join(seq1_raw)))
    seq2 = list(filter(lambda char: char != '\n',''.join(seq2_raw)))

    # check that length of sequences match
    if(len(seq1) != len(seq2)):
        print("sequences have different length!")
        print(len(seq1),' ',len(seq2))
        print(seq1_raw,'\n',seq2_raw)
        return 1

    # initialize frequency matrix
    freq = {'AA':0, 'AC':0, 'AG':0, 'AT':0, 'CA':0, 'CC':0, 'CG':0, 'CT':0,\
        'GA':0, 'GC':0, 'GG':0, 'GT':0, 'TA':0, 'TC':0, 'TG':0, 'TT':0}

    # count base transitions from seq1 to seq2
    for i in range(len(seq1)):
        freq[seq1[i]+seq2[i]] += 1

    return(list(freq.values()))
