// nnetbin/nnet-train-lstm-streams.cc

// Copyright 2015-2016  Brno University of Technology (Author: Karel Vesely)
//           2014  Jiayu DU (Jerry), Wei Li

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include <numeric>

#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-randomizer.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "cudamatrix/cu-device.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet1;
  typedef kaldi::int32 int32;

  try {
    const char *usage =
        "Perform one iteration of LSTM training by Stochastic Gradient Descent.\n"
        "The training targets are pdf-posteriors, usually prepared by ali-to-post.\n"
        "The updates are per-utterance.\n"
        "\n"
        "Usage: nnet-train-lstm-streams [options] "
          "<feature-rspecifier> <targets-rspecifier> <model-in> [<model-out>]\n"
        "e.g.: nnet-train-lstm-streams scp:feature.scp ark:posterior.ark nnet.init nnet.iter1\n";

    ParseOptions po(usage);

    NnetTrainOptions trn_opts;
    trn_opts.Register(&po);

    bool binary = true;
    po.Register("binary", &binary, "Write output in binary mode");

    bool crossvalidate = false;
    po.Register("cross-validate", &crossvalidate,
        "Perform cross-validation (don't back-propagate)");

    /*std::string feature_transform;
    po.Register("feature-transform", &feature_transform,
        "Feature transform in Nnet format");*/

    std::string objective_function = "xent";
    po.Register("objective-function", &objective_function,
        "Objective function : xent|mse");

    /*
    int32 length_tolerance = 5;
    po.Register("length-tolerance", &length_tolerance,
      "Allowed length difference of features/targets (frames)");

    std::string frame_weights;
    po.Register("frame-weights", &frame_weights,
      "Per-frame weights to scale gradients (frame selection/weighting).");
    */

    std::string use_gpu="yes";
    po.Register("use-gpu", &use_gpu,
        "yes|no|optional, only has effect if compiled with CUDA");

    // <shr>
    int32 targets_delay = 5;
    po.Register("targets-delay", &targets_delay, "[LSTM] BPTT targets delay");

    int32 num_stream = 4;
    po.Register("num-stream", &num_stream, "[LSTM] BPTT multi-stream training");
 
    double frame_limit = 1000000;
    po.Register("frame-limit", &frame_limit, "Max number of frames to be processed");
    // </shr>

    //// Add dummy option for compatibility with default scheduler,
    bool randomize = false;
    po.Register("randomize", &randomize,
        "Dummy, for compatibility with 'steps/nnet/train_scheduler.sh'");
    ////

    po.Read(argc, argv);

    if (po.NumArgs() != 5 + (crossvalidate ? 0 : 2)) {
      po.PrintUsage();
      exit(1);
    }

    std::string feature_rspecifier = po.GetArg(1),
      ce_rspecifier = po.GetArg(2),
      targets_rspecifier = po.GetArg(3),
      model1_filename = po.GetArg(4),
      model2_filename = po.GetArg(5);

    std::string target_model1_filename, target_model2_filename;
    if (!crossvalidate) {
      target_model1_filename = po.GetArg(6);
      target_model2_filename = po.GetArg(7);
    }
// TODO
    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;

#if HAVE_CUDA == 1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    /*Nnet nnet_transf;
    if (feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }*/

    Nnet nnet1, nnet2;
    nnet1.Read(model1_filename);
    nnet1.SetTrainOptions(trn_opts);
    nnet2.Read(model2_filename);
    nnet2.SetTrainOptions(trn_opts);

    kaldi::int64 total_frames = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    SequentialBaseFloatMatrixReader ce_reader(ce_rspecifier);
    //RandomAccessPosteriorReader target_reader(targets_rspecifier); // shr
    RandomAccessInt32VectorReader targets_reader(targets_rspecifier);

    /*
    RandomAccessBaseFloatVectorReader weights_reader;
    if (frame_weights != "") {
      weights_reader.Open(frame_weights);
    }
    */

    Xent xent;
    Mse mse;

    Timer time;
    KALDI_LOG << (crossvalidate ? "CROSS-VALIDATION" : "TRAINING")
              << " STARTED";

    int32 num_done = 0,
          num_no_tgt_mat = 0,
          num_other_error = 0;

   std::vector< Matrix<BaseFloat> > e_utt(num_stream);  // Feature matrix of every utterance, e.g., <begin> x x x x x <end>
   std::vector< Matrix<BaseFloat> > lm_utt(num_stream);  // Feature matrix of every utterance, e.g., <begin> x x x x x
   Matrix<BaseFloat> ce = ce_reader.Value();  // char-embeding matrix
   std::vector< std::vector<int32> > labels_utt(num_stream);  // Label vector of every utterance, e.g., x x x x x <end>
   int32 feat_e_dim = nnet1.InputDim();
   int32 feat_lm_dim = nnet2.InputDim();
   int32 out_lm_dim = nnet2.OutputDim();
   int32 out_e_dim = nnet1.OutputDim();
   CuMatrix<BaseFloat> e_nnet_out, lm_nnet_out, obj_diff, in_lm_diff, e_diff;  // NOTE: add parts. TODO
   Matrix<BaseFloat> e_mat, lm_mat;
   while(1)
   {
     std::vector<int32> frame_num_utt;
     int32 max_frame_num = 0, sequence_index=0;
     for ( ; !feature_reader.Done(); feature_reader.Next())
     {
       std::string utt = feature_reader.Key();
       //KALDI_LOG << "utt is: " << utt;
       if (!targets_reader.HasKey(utt)) 
       {
         KALDI_WARN << utt << ", missing targets";
         num_no_tgt_mat++;
         continue;
       }
       // get the e-matrix and lm-matrix form source-data
       Matrix<BaseFloat> mat = feature_reader.Value(); // in fact, the input matrix is a vector that mapped to rows in ce-matrix 
       e_mat.Resize(mat.NumRows(), feat_e_dim);
       lm_mat.Resize(mat.NumRows()-1, feat_e_dim);
       for (int32 t = 0; t < mat.NumRows(); t++) e_mat.Row(t).CopyFromVec(ce.Row( (int)mat(t, 0) ));
       lm_mat.CopyFromMat(e_mat.RowRange(0, lm_mat.NumRows()));
       
       // get targets and check that the length matches,
       std::vector<int32> targets = targets_reader.Value(utt); // target
       /*if (lm_mat.NumRows() != targets.size()) {
         KALDI_WARN << utt
           << ", length miss-match between feats and targets, skipping";
         num_other_error++;
         feature_reader.Next();
         continue;
       }*/
       
       // write e-matrix, lm-matrix and labels to streams
       if (max_frame_num < e_mat.NumRows()) max_frame_num = e_mat.NumRows();
       e_utt[sequence_index] = e_mat;
       lm_utt[sequence_index] = lm_mat;
       labels_utt[sequence_index] = targets;
       frame_num_utt.push_back(e_mat.NumRows());
       sequence_index ++;

       // If the total number of frames reaches frame_limit, then stop adding more sequences, regardless of whether the number of utterances reaches num_stream or not.
       if (frame_num_utt.size() == num_stream || frame_num_utt.size() * max_frame_num > frame_limit )
       {
         feature_reader.Next();
         break;
       }
     } // end loop for, that the streams-data is prepared!

     int32 cur_sequence_num = frame_num_utt.size();
     std::vector<int32> new_utt_flags(cur_sequence_num, 1);
     //Create the final feature matrix. Every utterance is padded to the max length within this group of utterances
     int32 max_frame_num_lm = max_frame_num - 1;
     Matrix<BaseFloat> e_mat_streams, lm_mat_streams, target_host; 
     e_mat_streams.Resize(cur_sequence_num * max_frame_num, feat_e_dim, kSetZero);
     lm_mat_streams.Resize(cur_sequence_num * max_frame_num_lm, feat_e_dim, kSetZero);
     target_host.Resize(cur_sequence_num * max_frame_num_lm, out_lm_dim, kSetZero);
     Vector<BaseFloat> frame_mask(cur_sequence_num * max_frame_num_lm, kSetZero);
     
     for (int32 s = 0; s < cur_sequence_num; s++)
     {
       Matrix<BaseFloat> e_mat_tmp = e_utt[s];
       Matrix<BaseFloat> lm_mat_tmp = lm_utt[s];   
       for (int32 r = 0; r < frame_num_utt[s]; r++)   
       {
         e_mat_streams.Row(r*cur_sequence_num + s).CopyFromVec(e_mat_tmp.Row(r));
         if ( r < frame_num_utt[s] - 1){
           lm_mat_streams.Row(r*cur_sequence_num + s).CopyFromVec(lm_mat_tmp.Row(r));           
         }
       }       
     }
     
     // for streams with new utterance, history states need to be reset   
     nnet1.ResetLstmSeq(new_utt_flags);
     nnet2.ResetLstmSeq(new_utt_flags); 
     //nnet1.SetSeqLengths(frame_num_utt);
     //nnet2.SetSeqLengths(frame_num_utt); 

     // forward pass nnet1 & nnet2
     // first, nnet1
     nnet1.Propagate(CuMatrix<BaseFloat>(e_mat_streams), &e_nnet_out);
     // second, extract the seg. form e_nnet_out matrix, and then add to lm_mat_streams(first stream)
     Vector<BaseFloat> last_row_utt(cur_sequence_num);
     for (int32 i = 0; i < cur_sequence_num; i++){
       last_row_utt(i) = (frame_num_utt[i] - 1) * cur_sequence_num + i ;
       lm_mat_streams.Row(i).CopyFromVec(e_nnet_out.Row( (int)last_row_utt(i) ));
     }
     
     for (int32 i = 0; i < cur_sequence_num; i++){
       std::vector<int32> targets_tmp = labels_utt[i];
       for (int32 j = 0; j < max_frame_num_lm; j++){   
         if ( j < frame_num_utt[i] - 1 ){
           target_host(j*cur_sequence_num + i, targets_tmp[j]) = 1;
           frame_mask(j*cur_sequence_num + i) = 1;
         }
         else{
           target_host(j*cur_sequence_num + i, targets_tmp[frame_num_utt[i] - 2]) = 1;
         }
       }
     }
     
     nnet2.Propagate(CuMatrix<BaseFloat>(lm_mat_streams), &lm_nnet_out);
     // cross-val
     xent.Eval(frame_mask, lm_nnet_out, CuMatrix<BaseFloat>(target_host), &obj_diff);
     // Backward-pass
     if (!crossvalidate)
     {
       nnet2.Backpropagate(obj_diff, &in_lm_diff);
       e_diff.Resize(e_nnet_out.NumRows(), e_nnet_out.NumCols(), kSetZero);
       for (int32 s = 0; s < cur_sequence_num; s++){
         e_diff.Row(last_row_utt(s)).CopyFromVec(in_lm_diff.Row(s));       
       }
       nnet1.Backpropagate(e_diff, NULL);

     }

     // 1st minibatch : show what happens in network,
     if (total_frames == 0) {
       KALDI_VLOG(1) << "### After " << total_frames << " frames,";
       KALDI_VLOG(1) << nnet1.InfoPropagate();
       KALDI_VLOG(1) << nnet2.InfoPropagate();
       if (!crossvalidate) {
         KALDI_VLOG(1) << nnet1.InfoBackPropagate();
         KALDI_VLOG(1) << nnet1.InfoGradient();
         KALDI_VLOG(1) << nnet2.InfoBackPropagate();
         KALDI_VLOG(1) << nnet2.InfoGradient();
       }
     }
 
     // VERBOSE LOG
     // monitor the NN training (--verbose=2),
     if (kaldi::g_kaldi_verbose_level >= 2) {
       static int32 counter = 0;
       counter += frame_mask.Sum();
       // print every 50k frames,
       if (counter >= 50000) {
         KALDI_VLOG(2) << "### After " << total_frames << " frames,";
         KALDI_VLOG(2) << nnet1.InfoPropagate();
         KALDI_VLOG(2) << nnet2.InfoPropagate();
         if (!crossvalidate) {
           KALDI_VLOG(2) << nnet1.InfoBackPropagate();
           KALDI_VLOG(2) << nnet1.InfoGradient();
           KALDI_VLOG(2) << nnet2.InfoBackPropagate();
           KALDI_VLOG(2) << nnet2.InfoGradient();
         }
         counter = 0;
       }
     }
     
     total_frames += frame_mask.Sum();
     num_done += cur_sequence_num;

     {  // do this every 25000 uttearnces,
       static int32 utt_counter = 0;
       utt_counter += cur_sequence_num;
       if (utt_counter >= 25000) {
         utt_counter = 0;
         // report speed,
         double time_now = time.Elapsed();
         KALDI_VLOG(1) << "After " << num_done << " utterances: "
           << "time elapsed = " << time_now / 60 << " min; "
           << "processed " << total_frames / time_now << " frames per sec.";
#if HAVE_CUDA == 1
         // check that GPU computes accurately,
         CuDevice::Instantiate().CheckGpuHealth();
#endif
       }
     }


     if (feature_reader.Done()) break; // end loop of while(1)
   }


    // after last minibatch : show what happens in network,
    KALDI_VLOG(1) << "### After " << total_frames << " frames,";
    KALDI_VLOG(1) << nnet1.InfoPropagate();
    KALDI_VLOG(1) << nnet2.InfoPropagate();
    if (!crossvalidate) {
      KALDI_VLOG(1) << nnet1.InfoBackPropagate();
      KALDI_VLOG(1) << nnet1.InfoGradient();
      KALDI_VLOG(1) << nnet2.InfoBackPropagate();
      KALDI_VLOG(1) << nnet2.InfoGradient();
    }

    if (!crossvalidate) {
      nnet1.Write(target_model1_filename, binary);
      nnet2.Write(target_model2_filename, binary);
    }

    KALDI_LOG << "Done " << num_done << " files, "
      << num_no_tgt_mat << " with no tgt_mats, "
      << "[" << (crossvalidate ? "CROSS-VALIDATION" : "TRAINING")
      << ", " << (randomize ? "RANDOMIZED" : "NOT-RANDOMIZED")
      << ", " << time.Elapsed() / 60 << " min, processing "
      << total_frames / time.Elapsed() << " frames per sec.]";

    KALDI_LOG << xent.Report();

#if HAVE_CUDA == 1
    CuDevice::Instantiate().PrintProfile();
#endif

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
