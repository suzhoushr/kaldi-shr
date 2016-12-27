#!/bin/bash

# this script is written by SHR, 2016.6
# this script will produce a word embedding file
. ./path.sh

if [ $# != 1 ]; then
   echo "Usage: $0 <data-dir>"
   echo " e.g.: $0 ./data"
   exit 1;
fi

pic=$1
#model=$2 shr_cld_ok/nnet/final.nnet shr_cld_small_model/nnet/test.nnet
#dict=$3

./shr_scripts/chw/demo/img2condata.py "$pic" "demo" | copy-feats ark:- ark,scp:./shr_scripts/chw/demo/demo.ark,./shr_scripts/chw/demo/demo.scp
chw-forward --use-gpu="yes" ./exp_chw/shr_cld_small_model/nnet/test.nnet ark:./shr_scripts/chw/demo/demo.ark ark:- | copy-feats ark:- ark,t:./shr_scripts/chw/demo/demo.forward.txt

./shr_scripts/chw/demo/forward.num2char.py ./shr_scripts/chw/demo/demo.forward.txt /home/lab/kaldi-shr/egs/shr_work/s5/data_chw/char_without_rare_num























