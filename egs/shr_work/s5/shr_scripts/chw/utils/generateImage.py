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


f_data = open(sys.argv[1],'r')

def genline(text, font, filename):
    '''
    generate one line
    '''
    add_h = 0
    add_w = 0
    w0, h0 = font.getsize(text)
    if h0 <= 64:
        h = 64
        #add_h = h -h0
        w = w0
    else:
        h = h0
        w = w0
        #print filename
    #add_w = int(round(1.0*w/h*add_h)) + 1
    #print str(w) + ' ' + str(h)
    #h += add_h
    #w += add_w
     
    #print str(w_new) + ' ' + str(h_new)   
    image = Image.new('RGB', (w, h), 'white')
    brush = ImageDraw.Draw(image)
    brush.text((0, 0), text, font=font, fill=(0, 0, 0))

    h_new = 64
    w_new = int(round(1.0 * h_new / (h+add_h) * (w+add_w))) + 1
    image_new = image.resize((w_new, h_new), Image.ANTIALIAS)
    #image_new = image
    #brush = ImageDraw.Draw(image_new)
    #brush.text((10, 8), text, font=font, fill=(0, 0, 0))
    image_new.save(filename + '.png')
    with open(filename + '.txt', 'w') as f:
        f.write(text.encode('utf-8'))
        f.close()

if __name__ == '__main__':
    if not os.path.isdir('./pics/'):
        os.mkdir('./pics/')
    fontname = './data/fonts/'
    font_id1 = random.randint(1,10)
    font_id2 = random.randint(1,10)
    while font_id1 == font_id2:
        font_id2 = random.randint(1,10)
    fontsize = 60
    font1 = ImageFont.truetype(fontname + str(font_id1) + '.ttf', fontsize)
    font2 = ImageFont.truetype(fontname + str(font_id2) + '.ttf', fontsize)
    i = 0
    for line in f_data.readlines():
        line = line.replace('\n','').strip()
        text = line.decode('utf-8')
        w1, h1 = font1.getsize(text)
        w2, h2 = font2.getsize(text)
        if h1 == 0 or w1 == 0 or h2 == 0 or w2 == 0:
            continue
        #print text
        filename = './pics/' + str(i + 1)
        genline(text, font1, filename)
        i += 1
        filename = './pics/' + str(i + 1)
        genline(text, font2, filename)
        i += 1
    f_data.close()

