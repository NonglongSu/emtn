#!/usr/bin/python3

import argparse
import re
import TN

def tn(args):
    if(args.input == 'multinomial'):
        TN.EM_TN(tolerance = args.tolerance, json_out = args.output, \
            no_print = args.no_print, test = args.test)
    elif(re.search('.dawg$', args.input[0]) != None):
        print('Use of dawg is not currently supported.\nPlease run dawg independently and input fasta file.')
    elif(re.search('.fasta$',args.input[0]) != None):
        TN.EM_TN(tolerance = args.tolerance, json_out = args.output[0], \
            no_print = args.no_print, N_counts = args.input[0], test = args.test)
    else:
        print('Dawg or fasta file input parsing failed.')

def hky(args):
    print("not ready")

def f84(args):
    print("not ready")

def k2p(args):
    print("not ready")

def jc(args):
    print("not ready")

def run_EM(args):
    return {'TN':tn,'HKY':hky,'F84':f84,'K2P':k2p,'JC':jc}.get(args.model)(args)


def main():
    # define argparse
    parser = argparse.ArgumentParser(description = 'Estimate model parameters')
    parser.add_argument('model', choices = ['TN','HKY','F84','K2P','JC'], help =
        'Select DNA model', default = 'TN')
    parser.add_argument('-n','--no-print', action = 'store_true', help =
        'Prevent printing of values every iteration.', default = False)
    parser.add_argument('--tolerance',type = float, default = 1e-10, help =
        'Tolerance for conversion')
    parser.add_argument('-i','--input', default = 'multinomial', \
        help = 'Input fasta or dawg file. If not input counts generated via multinomial distribution.',\
        nargs = 1)
    parser.add_argument('-o','--output', help = 'Json output file name',
        nargs = 1, default = 'out.json')
    parser.add_argument('-t','--test', default = False, help = 'Turn on testing\
        outputs.', action = 'store_true')

    args = parser.parse_args()
    run_EM(args)

if __name__ == '__main__':
    main()
