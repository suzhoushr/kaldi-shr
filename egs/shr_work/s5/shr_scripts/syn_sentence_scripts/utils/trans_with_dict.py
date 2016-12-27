#!/usr/bin/env python
#encoding=utf-8
# Copyright 2016 SHR SHGX


import sys
import string
import re
reload(sys)
sys.setdefaultencoding( "utf-8" )

if __name__ == '__main__':

    path_dict = sys.argv[1]
    path_file = sys.argv[2]
    split_sym = sys.argv[3]
    f_file = open(path_file,'r')
    d_file = open(path_dict,'r')
    dict_ori_list = []
    for line in d_file.readlines():
        line = line.replace('\n','').strip()
        dict_line = line.split(' ')[0]
        dict_ori_list.extend([dict_line.encode('utf8')])
    for line in f_file.readlines():
        out_line = ''
        line = line.replace('\n','').strip()
        temp_splits = line.split(split_sym)
        for n in range(0, len(temp_splits)):
            char_split = temp_splits[n].encode('utf8')
            if char_split in dict_ori_list: 
                out_line += char_split + split_sym
            else:
                out_line += '<UNK>' + split_sym 
        print out_line.strip()
