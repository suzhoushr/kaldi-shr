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
    path_file = sys.argv[2]
    func = sys.argv[3]
    flag_label = sys.argv[4]

    f_file = open(path_file,'r')
    we_file = open(path_we,'r')
    we_list = {}
    for line in we_file.readlines():
        line = line.replace('\n','').strip()
        word_embeding = line.split(' ')
        word = word_embeding[0]
        we_list[word] = word_embeding[1]
    we_file.close()     

    index = 0
    if flag_label == "yes":
      for line in f_file.readlines():
        index += 1
        out_line = func + '_' + str(index) + ' '
        line = line.replace('\n','').replace('\t','').strip()
        seq_list = line.split(' ')
            
        for n in range(1, len(seq_list)):
          try:
            out_line += we_list[seq_list[n]] + ' '
          except Exception:
            out_line += we_list['<UNK>'] + ' '           
        print out_line.strip()
    else:
      for line in f_file.readlines():
        index += 1
        out_line = func + '_' + str(index) + '  ['
        print out_line
        line = line.replace('\n','').replace('\t','').strip()
        seq_list = line.split(' ')             
        for n in range(0, len(seq_list)):
          out_line = '  '          
          try:
            out_line += we_list[seq_list[n]]
          except Exception:
            out_line += we_list['<UNK>']
          if n == len(seq_list) - 1:
            out_line += ' ]'            
          print out_line
    #print 'total error is: ' + str(error) + ' ' + info
    f_file.close()
