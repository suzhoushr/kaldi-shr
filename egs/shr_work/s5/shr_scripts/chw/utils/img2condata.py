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

f_hw = open(sys.argv[1], 'r')
func = sys.argv[2]
index = 0
for line in f_hw.readlines():
    index += 1
    line = line.replace('\n','').strip()
    path_pic = line + '.png'
    #path_txt = line + '.txt'
    image_rgb = Image.open(path_pic) 
    image_gray = image_rgb.convert('L')
    w, h = image_gray.size
    print func + '_' + str(index) + '  [' 
    flag = 1
    iw = 0
    win = 8
    step = 5
    while flag == 1:
        out = '  '
        for n in range(win):            
            for ih in range(h):
                if iw+n < w:
                    out += str((255 - image_gray.getpixel((iw+n, ih)))) + ' '
                else:
                    out += str(0) + ' '        
        if iw + step < w:
            iw += step
        else:  
            out += ']'          
            flag = 0
        print out.rstrip()
f_hw.close()
