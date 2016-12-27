#!/bin/bash

# this script is written by SHR, 2016.6
# this script will produce a word embedding file
. ./path.sh

if [ $# != 1 ]; then
   echo "Usage: $0 <data-dir>"
   echo " e.g.: $0 ./data"
   exit 1;
fi

dir=$1
sort_by_len=false
cur_path=`pwd`/shr_scripts/syn_sentence_scripts/utils
w2v_path=~/tools/w2v/trunk
dict_max_size=5000
stage=0

if [ $stage -le 0 ]; then
## split the ori. data into two parts: data and data_trans 
  [ -f $dir/data.ori.txt ] || echo "no original data" 
  [ -f $dir/data.alice.char.txt ] || cat $dir/data.ori.txt | awk -F "\t" '{print $1}' | perl -e 'use encoding utf8; while(<>){ chop; $str=""; foreach $p (split("", $_)) {$str="$str$p<->"}; print "$str\n";}' > $dir/tmp.txt && $cur_path/processing_data_unit.py $dir/tmp.txt "<->" " " | cat - | awk '{print "<begin> "$0" <end>"}' > $dir/data.alice.char.txt && rm $dir/tmp.txt
  [ -f $dir/data.bob.char.txt ] || cat $dir/data.ori.txt | awk -F "\t" '{print $2}' | perl -e 'use encoding utf8; while(<>){ chop; $str=""; foreach $p (split("", $_)) {$str="$str$p<->"}; print "$str\n";}' > $dir/tmp.txt && $cur_path/processing_data_unit.py $dir/tmp.txt "<->" " " | cat - | awk '{print "<begin> "$0" <end>"}' > $dir/data.bob.char.txt && rm $dir/tmp.txt
fi
## prepare data to produce dict and char-embeding  | sort -R
# char-dict and char-embeding
data_file=$dir/data.alice.char.txt
we_file=$dir/char.embeding.txt
dict_file=$dir/dict.txt
if [ $stage -le 1 ]; then
  time $w2v_path/word2vec -train $data_file -output $we_file -cbow 1 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 0 -iter 15 -min-count 1
  dict_size=`head -n 1 $we_file | awk '{print $1}'`
  sed -i '1, 2d' $we_file
  if [ $dict_size -gt $dict_max_size ]; then
    $cur_path/prep_dict.py $we_file > $dir/tmp.dict.txt
    cat $dir/tmp.dict.txt | head -n $dict_max_size > $dir/tmp.txt
    rm $dir/tmp.dict.txt
    (echo '<UNK>') | cat - $dir/tmp.txt > $dict_file
    rm $dir/tmp.txt $we_file
    $cur_path/trans_with_dict.py $dict_file $dir/data.alice.char.txt " " > $dir/data.alice.char.new.txt
    rm $dict_file
    mv $data_file $dir/data.alice.char.ori.txt
    mv $dir/data.alice.char.new.txt $data_file
    time $w2v_path/word2vec -train $data_file -output $we_file -cbow 1 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-4 -threads 20 -binary 0 -iter 15 -min-count 1
    sed -i '1, 2d' $we_file
    $cur_path/prep_dict.py $we_file > $dict_file
  else
    $cur_path/prep_dict.py $we_file > $dict_file
  fi
  $cur_path/prep_char_embending_matrix.py $we_file "char_embeding_matrix" | copy-feats ark:- ark,scp:$dir/ce.ark,$dir/ce.scp
  mv $dir/data.bob.char.txt $dir/data.bob.char.ori.txt
  $cur_path/trans_with_dict.py $dict_file $dir/data.bob.char.ori.txt " " > $dir/data.bob.char.txt
fi 

if [ $stage -le 2 ]; then
  ## prepare train and test data
  len_data=`cat $data_file | wc -l`
  len_cv=`expr $len_data / 100`
  len_tr=`expr $len_data - $len_cv`
  cat $data_file | head -n $len_cv > $dir/data.alice.cv.txt
  cat $data_file | tail -n $len_tr > $dir/data.alice.tr.txt
  cat $dir/data.bob.char.txt | head -n $len_cv > $dir/data.bob.cv.txt
  cat $dir/data.bob.char.txt | tail -n $len_tr > $dir/data.bob.tr.txt
  $cur_path/prep_num_data.py $dict_file $dir/data.alice.tr.txt "train" "no" | copy-feats ark:- ark,scp:$dir/data.alice.tr.ark,$dir/data.alice.tr.scp
  $cur_path/prep_num_data.py $dict_file $dir/data.alice.cv.txt "test" "no" | copy-feats ark:- ark,scp:$dir/data.alice.cv.ark,$dir/data.alice.cv.scp
fi

if [ $stage -le 3 ]; then
  ## label
  $cur_path/prep_num_data.py $dict_file $dir/data.bob.tr.txt "train" "yes" | gzip -c - > $dir/label.bob.tr.gz
  $cur_path/prep_num_data.py $dict_file $dir/data.bob.cv.txt "test" "yes" | gzip -c - > $dir/label.bob.cv.gz
fi

if [ $stage -le 4 ]; then
  ## sort or rand the scp
  if $sort_by_len; then
    feat-to-len scp:$dir/data.alice.tr.scp ark,t:- | awk '{print $2}' > $dir/len.tr.tmp || exit 1;
    paste -d " " $dir/data.alice.tr.scp $dir/len.tr.tmp | sort -k3 -n - | awk '{print $1 " " $2}' > $dir/tr.scp || exit 1;
    feat-to-len scp:$dir/data.alice.cv.scp ark,t:- | awk '{print $2}' > $dir/len.cv.tmp || exit 1;
    paste -d " " $dir/data.alice.cv.scp $dir/len.cv.tmp | sort -k3 -n - | awk '{print $1 " " $2}' > $dir/cv.scp || exit 1;
    #rm -f $dir/len.tmp
  else
    cat $dir/data.alice.tr.scp | utils/shuffle_list.pl --srand ${seed:-777} > $dir/tr.scp
    cat $dir/data.alice.cv.scp | utils/shuffle_list.pl --srand ${seed:-777} > $dir/cv.scp
  fi
fi

























