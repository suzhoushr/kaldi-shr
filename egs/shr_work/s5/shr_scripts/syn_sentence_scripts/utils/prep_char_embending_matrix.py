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
    func = sys.argv[2]

    we_file = open(path_we,'r')
    count_lines = 0
    len_we = len(we_file.readlines())
    we_file = open(path_we,'r')
    print func + '  ['
    for line in we_file.readlines():
      line = line.replace('\n','').strip()
      count_lines += 1
      word_embeding = line.split(' ')
      out_line = '  '
      for n in range(1, len(word_embeding)):
        out_line += word_embeding[n] + ' '
      out_line = out_line.strip()
      if count_lines == len_we:
        print out_line + ' ]'
      else:
        print out_line
    we_file.close()     

