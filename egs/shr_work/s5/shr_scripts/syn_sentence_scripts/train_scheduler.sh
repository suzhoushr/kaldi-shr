#!/bin/bash

# Copyright 2012-2015  Brno University of Technology (author: Karel Vesely)
# Apache 2.0

# Schedules epochs and controls learning rate during the neural network training

# Begin configuration.

# training options,
learn_rate=0.008
momentum=0.9
l1_penalty=0
l2_penalty=0

# data processing,
train_tool="nnet-train-syn-sentence" #"nnet-train-ars-elm"
train_tool_opts="--targets-delay=0 --num-stream=300"

# learn rate scheduling,
max_iters=40
min_iters=0 # keep training, disable weight rejection, start learn-rate halving as usual,
keep_lr_iters=0 # fix learning rate for N initial epochs, disable weight rejection,
start_halving_impr=0.01
end_halving_impr=0.001
halving_factor=0.5

# misc,
verbose=1
frame_weights=
 
# End configuration.

echo "$0 $@"  # Print the command line for logging
[ -f path.sh ] && . ./path.sh; 

. parse_options.sh || exit 1;

set -euo pipefail

if [ $# != 7 ]; then
   echo "Usage: $0 <mlp-init> <feats-tr> <feats-cv> <labels-tr> <labels-cv> <exp-dir>"
   echo " e.g.: $0 0.nnet scp:train.scp scp:cv.scp ark:labels_tr.ark ark:labels_cv.ark exp/dnn1"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>  # config containing options"
   exit 1;
fi

dir_nnet=$1
tr=$2
cv=$3
ce=$4
labels_tr=$5
labels_cv=$6
dir=$7

mlp1_init=$dir_nnet/nnet1.iter0
mlp2_init=$dir_nnet/nnet2.iter0

[ ! -d $dir ] && mkdir $dir
[ -e $dir/final.nnet1 ] && echo "'$dir/final.nnet1' exists, skipping training" && exit 0



##############################
# start training

# choose mlp to start with,
mlp1_best=$mlp1_init
mlp2_best=$mlp2_init
mlp1_base=${mlp1_init##*/}; mlp1_base=${mlp1_base%.*}
mlp2_base=${mlp2_init##*/}; mlp2_base=${mlp2_base%.*}

# optionally resume training from the best epoch, using saved learning-rate,
[ -e $dir/.mlp1_best ] && mlp1_best=$(cat $dir/.mlp1_best)
[ -e $dir/.mlp2_best ] && mlp2_best=$(cat $dir/.mlp2_best)
[ -e $dir/.learn_rate ] && learn_rate=$(cat $dir/.learn_rate)

# cross-validation on original network,     
log=$dir/log/iter00.initial.log; hostname>$log
$train_tool --cross-validate=true --randomize=false --verbose=$verbose $train_tool_opts \
  ${frame_weights:+ "--frame-weights=$frame_weights"} \
  "$cv" "$ce" "$labels_cv" $mlp1_best $mlp2_best \
  2>> $log

loss=$(cat $dir/log/iter00.initial.log | grep "AvgLoss:" | tail -n 1 | awk '{ print $4; }')
loss_type=$(cat $dir/log/iter00.initial.log | grep "AvgLoss:" | tail -n 1 | awk '{ print $5; }')
echo "CROSSVAL PRERUN AVG.LOSS $(printf "%.4f" $loss) $loss_type"

# resume lr-halving,
halving=0
[ -e $dir/.halving ] && halving=$(cat $dir/.halving)

# training,
for iter in $(seq -w $max_iters); do
  echo -n "ITERATION $iter: "
  mlp1_next=$dir/nnet/${mlp1_base}_iter${iter}
  mlp2_next=$dir/nnet/${mlp2_base}_iter${iter}
  
  # skip iteration (epoch) if already done,
  [ -e $dir/.done_iter$iter ] && echo -n "skipping... " && ls $mlp1_next* && ls $mlp2_next* && continue 
  
  # training,
  log=$dir/log/iter${iter}.tr.log; hostname>$log
  $train_tool --cross-validate=false --randomize=true --verbose=$verbose $train_tool_opts \
    --learn-rate=$learn_rate --momentum=$momentum \
    --l1-penalty=$l1_penalty --l2-penalty=$l2_penalty \
    ${frame_weights:+ "--frame-weights=$frame_weights"} \
    "$tr" "$ce" "$labels_tr" $mlp1_best $mlp2_best $mlp1_next $mlp2_next \
    2>> $log || exit 1; 

  tr_loss=$(cat $dir/log/iter${iter}.tr.log | grep "AvgLoss:" | tail -n 1 | awk '{ print $4; }')
  echo -n "TRAIN AVG.LOSS $(printf "%.4f" $tr_loss), (lrate$(printf "%.6g" $learn_rate)), "
  
  # cross-validation,
  log=$dir/log/iter${iter}.cv.log; hostname>$log
  $train_tool --cross-validate=true --randomize=false --verbose=$verbose $train_tool_opts \
    ${frame_weights:+ "--frame-weights=$frame_weights"} \
    "$cv" "$ce" "$labels_cv" $mlp1_next $mlp2_next \
    2>>$log || exit 1;
  
  loss_new=$(cat $dir/log/iter${iter}.cv.log | grep "AvgLoss:" | tail -n 1 | awk '{ print $4; }')
  echo -n "CROSSVAL AVG.LOSS $(printf "%.4f" $loss_new), "

  # accept or reject?
  loss_prev=$loss
  if [ 1 == $(bc <<< "$loss_new < $loss") -o $iter -le $keep_lr_iters -o $iter -le $min_iters ]; then
    # accepting: the loss was better, or we had fixed learn-rate, or we had fixed epoch-number,
    loss=$loss_new
    mlp1_best=$dir/nnet/${mlp1_base}_iter${iter}_learnrate${learn_rate}_tr$(printf "%.4f" $tr_loss)_cv$(printf "%.4f" $loss_new)
    mlp2_best=$dir/nnet/${mlp2_base}_iter${iter}_learnrate${learn_rate}_tr$(printf "%.4f" $tr_loss)_cv$(printf "%.4f" $loss_new)
    [ $iter -le $min_iters ] && mlp1_best=${mlp1_best}_min-iters-$min_iters && mlp2_best=${mlp2_best}_min-iters-$min_iters 
    [ $iter -le $keep_lr_iters ] && mlp1_best=${mlp1_best}_keep-lr-iters-$keep_lr_iters && mlp2_best=${mlp2_best}_keep-lr-iters-$keep_lr_iters
    mv $mlp1_next $mlp1_best
    mv $mlp2_next $mlp2_best
    echo "nnet accepted ($(basename $mlp1_best))"
    echo "nnet accepted ($(basename $mlp2_best))"
    echo $mlp1_best > $dir/.mlp1_best
    echo $mlp2_best > $dir/.mlp2_best
  else
    # rejecting,
    mlp1_reject=$dir/nnet/${mlp1_base}_iter${iter}_learnrate${learn_rate}_tr$(printf "%.4f" $tr_loss)_cv$(printf "%.4f" $loss_new)_rejected
    mlp2_reject=$dir/nnet/${mlp2_base}_iter${iter}_learnrate${learn_rate}_tr$(printf "%.4f" $tr_loss)_cv$(printf "%.4f" $loss_new)_rejected
    mv $mlp1_next $mlp1_reject
    mv $mlp2_next $mlp2_reject
    echo "nnet rejected ($(basename $mlp1_reject))"
    echo "nnet rejected ($(basename $mlp2_reject))"
  fi

  # create .done file, the iteration (epoch) is completed,
  touch $dir/.done_iter$iter
  
  # continue with original learn-rate,
  [ $iter -le $keep_lr_iters ] && continue 

  # stopping criterion,
  rel_impr=$(bc <<< "scale=10; ($loss_prev-$loss)/$loss_prev")
  if [ 1 == $halving -a 1 == $(bc <<< "$rel_impr < $end_halving_impr") ]; then
    if [ $iter -le $min_iters ]; then
      echo we were supposed to finish, but we continue as min_iters : $min_iters
      continue
    fi
    echo finished, too small rel. improvement $rel_impr
    break
  fi

  # start learning-rate fade-out when improvement is low,
  if [ 1 == $(bc <<< "$rel_impr < $start_halving_impr") ]; then
    halving=1
    echo $halving >$dir/.halving
  fi
  
  # reduce the learning-rate,
  if [ 1 == $halving ]; then
    learn_rate=$(awk "BEGIN{print($learn_rate*$halving_factor)}")
    echo $learn_rate >$dir/.learn_rate
  fi
done

# select the best network,
if [ $mlp1_best != $mlp1_init ]; then 
  mlp1_final=${mlp1_best}_final_
  mlp2_final=${mlp2_best}_final_
  ( cd $dir/nnet; ln -s $(basename $mlp1_best) $(basename $mlp1_final); )
  ( cd $dir; ln -s nnet/$(basename $mlp1_final) final.nnet1; )
  ( cd $dir/nnet; ln -s $(basename $mlp2_best) $(basename $mlp2_final); )
  ( cd $dir; ln -s nnet/$(basename $mlp2_final) final.nnet2; )
  echo "Succeeded training the Neural Networks : $dir/final.nnet1, $dir/final.nnet2"
else
  "Error training neural network..."
  exit 1
fi

