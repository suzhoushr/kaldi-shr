#!/usr/bin/env python
#encoding=utf-8
# Copyright 2016 SHR SHGX


import sys
import jieba
import string
import re
reload(sys)
sys.setdefaultencoding( "utf-8" )

if __name__ == '__main__':

    path_we = sys.argv[1]
    we_file = open(path_we,'r')

    index = 0
    for line in we_file.readlines():
        line = line.replace('\n','').strip()
        word_embeding = line.split(' ')
        word = word_embeding[0]
        print word + ' ' + str(index)
        index += 1
    we_file.close() 
