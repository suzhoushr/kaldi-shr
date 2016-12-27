#!/usr/bin/env python

# Copyright 2015       Yajie Miao    (Carnegie Mellon University)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

# This python script converts the word-based transcripts into label sequences. The labels are
# represented by their indices. 

import sys

if __name__ == '__main__':

    len_file = sys.argv[1]
    f_len = open(len_file,'r')
    len_feats = {}
    for line in f_len.readlines():
        out_line = ''
        line = line.replace('\n','')
        splits = line.split(' ')
        utt = splits[0]
        out_line += utt	
        for n in range(0,int(splits[1])):
            out_line += ' ' + '0'
        print out_line
    f_len.close()