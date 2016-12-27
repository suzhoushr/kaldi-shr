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


#text = u"这是一段测试文本，test 123。"
 
#im = Image.new("RGB", (300, 50), (255, 255, 255))
#dr = ImageDraw.Draw(im)
#font = ImageFont.truetype(os.path.join("fonts", "yuhongliang.ttf"), 14)
 
#dr.text((10, 5), text, font=font, fill="#000000")
 
#im.show()
#im.save("t.png")

f_fwd_num = open(sys.argv[1], 'r')
f_dict = open(sys.argv[2], 'r')

char_dict = {}
for line in f_dict.readlines():
    line = line.replace('\n','').strip()
    splits = line.split(' ')
    char_dict[splits[1]] = splits[0]
f_dict.close()

line_num = 1
for line in f_fwd_num.readlines():
    if line_num == 1:
        line_num = 2
        continue
    line = line.replace('\n','').strip()
    splits = line.split(' ')
    if splits[0] == '0':
        print "oops, no result, maybe we made a mistake"
    else:
        out = ''
        for n in range(1, len(splits)-1):
            out += char_dict[splits[n]]
        print out
f_fwd_num.close()
