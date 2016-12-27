#!/usr/bin/env python
# coding: utf-8

import random
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

import sys
import os

reload(sys)
sys.setdefaultencoding( "utf-8" )

f_list = open(sys.argv[1], 'r')
f_dict = open(sys.argv[2], 'r')
func = sys.argv[3]

ch_dict = {}
for line in f_dict.readlines():
    line = line.replace('\n','').strip()
    splits = line.split(' ')
    ch_dict[splits[0].encode('utf-8')] = splits[1]
f_dict.close()





index = 0
for line in f_list.readlines():
    index += 1
    line = line.replace('\n','').strip()
    #path_pic = line + '.png'
    path_txt = line + '.txt'
    f_text = open(path_txt, 'r')
    for text in f_text.readlines():
        text = text.replace('\n','').strip()
        #print text
        out = func + '_' + str(index) + ' ' 
        for n in range(len(text)/3):
            out += str(0) + ' '            
        print out.strip()
        break
    f_text.close()
f_list.close()
