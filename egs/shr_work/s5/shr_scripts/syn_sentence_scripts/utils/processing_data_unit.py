#!/usr/bin/env python
#encoding=utf-8
# Copyright 2016 SHR SHGX


import sys
import string
import re
reload(sys)
sys.setdefaultencoding( "utf-8" )

# perl -e 'use encoding utf8; while(<>){ chop; $str=""; foreach $p (split("", $_)) {$str="$str$p<->"}; print "$str\n";}' > $dir/tmp.txt && $cur_path/processing_data_unit.py $dir/tmp.txt "<->" " "

if __name__ == '__main__':

    path_file = sys.argv[1]
    sym_split = sys.argv[2]
    sym_new_split = sys.argv[3]
    f_file = open(path_file,'r')
    en_list = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ"
    num_list = "1234567890１２３４５６７８９０."
    for line in f_file.readlines():
        line = line.replace('\n','').strip()
        splits = line.split(sym_split)
        #print len(splits)
        n = 0
        out_ = ''
        while n < len(splits) - 1:
          #print str(n) + ' ' +out_
          if en_list.find(splits[n]) > -1:
            out_ += splits[n]
            n += 1
            while n < len(splits) - 1 and en_list.find(splits[n]) > -1 :
              out_ += splits[n]
              n += 1
            out_ += sym_new_split
          elif num_list.find(splits[n]) > -1:
            out_ += splits[n]
            n += 1
            while n < len(splits) - 1 and num_list.find(splits[n]) > -1:
              out_ += splits[n]
              n += 1
            out_ += sym_new_split
          else:
            out_ += splits[n] + sym_new_split
            n += 1
        print out_.replace('   ',' ').strip(sym_new_split)

    f_file.close()



