#!/usr/bin/env python
# -*- coding:utf-8 -*- 
# Created by Roger on 2019/9/25
import sys


def main():
    tbf_file = sys.argv[1]
    plain_file = sys.argv[1] + '.plain'
    with open(plain_file, 'w') as output:
        for line in open(tbf_file):
            att = line.strip().split('\t')
            if len(att) == 7:
                att[5] = 'event'
                att[6] = 'Other'
            output.write('\t'.join(att) + '\n')


if __name__ == "__main__":
    main()
