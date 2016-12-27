#!/bin/bash

# Copyright 2012-2015  Brno University of Technology (author: Karel Vesely)
# Apache 2.0

# Begin configuration.

config=             # config, also forwarded to 'train_scheduler.sh',

# topology, initialization,
network_type=lstm    # select type of neural network (dnn,cnn1d,cnn2d,lstm),
hid_layers=4        # nr. of hidden layers (before sotfmax or bottleneck),
hid_dim=1024        # number of neurons per layer,
bn_dim=             # (optional) adds bottleneck and one more hidden layer to the NN,
dbn=                # (optional) prepend layers to the initialized NN,

proto_opts=         # adds options to 'make_nnet_proto.py',
cnn_proto_opts=     # adds options to 'make_cnn_proto.py',

nnet1_init=          # (optional) use this pre-initialized NN,
nnet2_init=          # (optional) use this pre-initialized NN,
nnet1_proto=         # (optional) use this NN prototype for initialization,
nnet2_proto=         # (optional) use this NN prototype for initialization,

# feature processing,
splice=5            # (default) splice features both-ways along time axis,
cmvn_opts=          # (optional) adds 'apply-cmvn' to input feature pipeline, see opts,
delta_opts=         # (optional) adds 'add-deltas' to input feature pipeline, see opts,
ivector=            # (optional) adds 'append-vector-to-feats', it's rx-filename,

feat_type=plain  
traps_dct_basis=11    # (feat_type=traps) nr. of DCT basis, 11 is good with splice=10,
transf=               # (feat_type=transf) import this linear tranform,
splice_after_transf=5 # (feat_type=transf) splice after the linear transform,

feature_transform_proto= # (optional) use this prototype for 'feature_transform',
feature_transform=  # (optional) directly use this 'feature_transform',
pytel_transform=    # (BUT) use external python transform,

# labels,
labels=            # (optional) specify non-default training targets,
                   # (targets need to be in posterior format, see 'ali-to-post', 'feat-to-post'),
num_tgt=           # (optional) specifiy number of NN outputs, to be used with 'labels=',

# training scheduler,
learn_rate=0.008   # initial learning rate,
scheduler_opts=    # options, passed to the training scheduler,
train_tool=        # optionally change the training tool,
train_tool_opts=   # options for the training tool,
frame_weights=     # per-frame weights for gradient weighting,

# data processing, misc.
copy_feats=true     # resave the train/cv features into /tmp (disabled by default),
copy_feats_tmproot=/tmp/kaldi.XXXX # sets tmproot for 'copy-feats',
seed=777            # seed value used for data-shuffling, nn-initialization, and training,
skip_cuda_check=false
begin_having=0

# End configuration.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh; 
. parse_options.sh || exit 1;

set -euo pipefail

if [ $# != 2 ]; then
   echo "Usage: $0 <data_dir> <exp-dir>"
   echo " e.g.: $0 data exp/mono_nnet"
   exit 1;
fi

data_dir=$1
dir=$2

mkdir -p $dir/{log,nnet}

# check files
for f in $data_dir/data.alice.tr.ark $data_dir/data.alice.cv.ark $data_dir/data.alice.tr.scp $data_dir/data.alice.cv.scp $data_dir/ce.ark $data_dir/ce.scp $data_dir/label.bob.tr.gz $data_dir/label.bob.cv.gz; do
  [ ! -f $f ] && echo "no such file $f" && exit 1;
done
cp $data_dir/tr.scp $dir/
cp $data_dir/cv.scp $dir/
cp $data_dir/label.*.gz $dir/
cp $data_dir/ce.scp $dir/

## Set up data  
tr="ark:copy-feats scp:$dir/tr.scp ark:- |"
cv="ark:copy-feats scp:$dir/cv.scp ark:- |"
ce="ark:copy-feats scp:$dir/ce.scp ark:- |"
## Set up labels  
labels_tr="ark:gunzip -c $dir/label.bob.tr.gz |"
labels_cv="ark:gunzip -c $dir/label.bob.cv.gz |"

## Initialize model parameters
# nnet1
if [ -z $nnet1_init ]; then 
   if [ ! -z $nnet1_proto ]; then
      if [ ! -f $dir/nnet/nnet1.iter0 ]; then
         echo "Initializing model1 as $dir/nnet/nnet1.iter0"
         nnet-initialize --binary=true $nnet1_proto $dir/nnet/nnet1.iter0 >& $dir/log/initialize_model1.log || exit 1;
      fi
   fi
else
   cp $nnet1_init $dir/nnet/nnet1.iter0
fi
# nnet2
if [ -z $nnet2_init ]; then 
   if [ ! -z $nnet2_proto ]; then
      if [ ! -f $dir/nnet/nnet2.iter0 ]; then
         echo "Initializing model2 as $dir/nnet/nnet2.iter0"
         nnet-initialize --binary=true $nnet2_proto $dir/nnet/nnet2.iter0 >& $dir/log/initialize_model2.log || exit 1;
      fi
   fi
else
   cp $nnet2_init $dir/nnet/nnet2.iter0
fi



###### TRAIN ######
echo
echo "# RUNNING THE NN-TRAINING SCHEDULER"
shr_scripts/syn_sentence_scripts/train_scheduler.sh \
  ${scheduler_opts} \
  ${train_tool:+ --train-tool "$train_tool"} \
  ${train_tool_opts:+ --train-tool-opts "$train_tool_opts"} \
  --learn-rate $learn_rate \
  --keep-lr-iters $begin_having \
  ${frame_weights:+ --frame-weights "$frame_weights"} \
  ${config:+ --config $config} \
  $dir/nnet "$tr" "$cv" "$ce" "$labels_tr" "$labels_cv" $dir || exit 1

echo "$0 successfuly finished.. $dir"
## output TODO
#nnet-ss-forward-shr --use-gpu="yes" --nnet-seclect=1 $dir/final.nnet1 $dir/final.nnet2 $dir/final.nnet3 "$en_cv" "$fr_cv" ark:- copy-feats ark:- ark,t:$dir/ss.output.txt
exit 0
