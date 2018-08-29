from __future__ import print_function
import os
import sys
import argparse
from random import shuffle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help="Input Data")
    args = parser.parse_args()

    filename = args.input_file
    f = open(filename, 'r')
    list_content = []
    for line in f:
        tmp = line.split(' ')
        if len(tmp) <= 1 or ':' in tmp[0]:
            print(line)
            sys.exit(1)
        tmp[0] = str( int(tmp[0]) + 1 )
        newline = ' '.join(tmp)
        newline = newline.replace('\n', '')
        list_content.append(newline)

    shuffle(list_content)
    n = len(list_content)
    n_tr = int(n*0.9)
    n_te = n - n_tr
    #split train \ test
    tr_content = list_content[0:n_tr]
    te_content = list_content[n_tr:]
    print("transform complete!")
    #save file
    g_tr = open("aloi_train", 'w')
    g_te = open("aloi_test", 'w')

    for line in tr_content:
        print(line, file = g_tr)

    for line in te_content:
        print(line, file = g_te)
