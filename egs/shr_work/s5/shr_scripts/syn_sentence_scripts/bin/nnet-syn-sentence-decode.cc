// nnetbin/nnet-forward.cc

// Copyright 2011-2013  Brno University of Technology (Author: Karel Vesely)

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

#include <limits>

#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "nnet/nnet-pdf-prior.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet1;
  try {
    const char *usage =
      "Perform forward pass through Neural Network.\n"
      "Usage: nnet-forward [options] <nnet1-in> <feature-rspecifier> <feature-wspecifier>\n"
      "e.g.: nnet-forward final.nnet ark:input.ark ark:output.ark\n";

    ParseOptions po(usage);

    PdfPriorOptions prior_opts;
    prior_opts.Register(&po);

    std::string use_gpu="no";
    po.Register("use-gpu", &use_gpu,
        "yes|no|optional, only has effect if compiled with CUDA");

    using namespace kaldi;
    using namespace kaldi::nnet1;
    typedef kaldi::int32 int32;

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string model1_filename = po.GetArg(1),
        model2_filename = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        ce_feature_rspecifier = po.GetArg(4),
        feature_wspecifier = po.GetArg(5);

    // Select the GPU
#if HAVE_CUDA == 1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    Nnet nnet1, nnet2;
    nnet1.Read(model1_filename);
    nnet2.Read(model2_filename);

    nnet1.SetDropoutRetention(1.0);
    nnet2.SetDropoutRetention(1.0);

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    SequentialBaseFloatMatrixReader ce_reader(ce_feature_rspecifier);
    BaseFloatMatrixWriter feature_writer(feature_wspecifier);

    CuMatrix<BaseFloat> e_nnet_out, lm_nnet_out;
    Matrix<BaseFloat> e_mat, lm_mat;

    Timer time;
    double time_now = 0;
    int32 num_done = 0;

    // main loop,
    Matrix<BaseFloat> ce = ce_reader.Value();  // char-embeding matrix( include <UNK> symbol )
    int32 feat_e_dim = nnet1.InputDim();
    int32 feat_lm_dim = nnet2.InputDim();
    
    for (; !feature_reader.Done(); feature_reader.Next()) {  
      
      bool is_end = false;

      // read
      Matrix<BaseFloat> mat = feature_reader.Value();   // load the input-num sequence
      e_mat.Resize(mat.NumRows(),feat_e_dim);
      lm_mat.Resize(1, feat_lm_dim);
      lm_mat.ColRange(feat_lm_dim-feat_e_dim, feat_e_dim).CopyFromMat(ce.RowRange(0,1));

      for (int32 i = 0; i < mat.NumRows(); i++)  e_mat.Row(i).CopyFromVec(ce.Row((int)mat(i,0)));

      std::string utt = feature_reader.Key();
      KALDI_VLOG(2) << "Processing utterance " << num_done+1
                    << ", " << utt
                    << ", " << e_mat.NumRows() << "frm";

      if (!KALDI_ISFINITE(e_mat.Sum())) {  // check there's no nan/inf,
        KALDI_ERR << "NaN or inf found in features for " << utt;
      }

      // forward pass nnet1(extracter), and extracting the feature vector v from utt
      std::vector<int32> new_utt_flags(1, 1);
      nnet1.ResetLstmStreams(new_utt_flags);
      nnet1.Feedforward(CuMatrix<BaseFloat>(e_mat), &e_nnet_out);
      // decode
      lm_mat.ColRange(0, feat_lm_dim - feat_e_dim).CopyFromMat( e_nnet_out.RowRange(mat.NumRows()-1, 1) );      
      std::vector<int32> index_seq;

      // # test      
      nnet2.ResetLstmStreams(new_utt_flags);
      while ( !is_end ) {
        //nnet2.ResetLstmStreams(new_utt_flags);
        nnet2.Feedforward(CuMatrix<BaseFloat>(lm_mat), &lm_nnet_out); // Propagate
        //new_utt_flags[0] = 1;
        //nnet2.ResetLstmStreams(new_utt_flags); 
        int32 max_index;
        BaseFloat max_value = 0;
        for (int32 i = 0; i < lm_nnet_out.NumCols(); i++)
          if ( max_value <= lm_nnet_out(lm_nnet_out.NumRows()-1,i) ) { max_value = lm_nnet_out(lm_nnet_out.NumRows()-1,i); max_index = i; }
        index_seq.push_back(max_index);
        //KALDI_LOG << "max index is: " << max_index;
        if ( max_index == 1 ) {
          is_end = true; 
        }
        else {
          //lm_mat.Resize(1, feat_lm_dim);
          lm_mat.ColRange(feat_lm_dim-feat_e_dim, feat_e_dim).CopyFromMat(ce.RowRange(max_index,1));
        }
        if (index_seq.size() > 30) is_end = true;
        
      }

      // # end test


      Matrix<BaseFloat> nnet_out_host(1, index_seq.size());
      for (int32 i =0; i < index_seq.size(); i++)
        nnet_out_host(0,i) = index_seq[i];
      

      // write,
      if (!KALDI_ISFINITE(nnet_out_host.Sum())) {  // check there's no nan/inf,
        KALDI_ERR << "NaN or inf found in final output nn-output for " << utt;
      }
      feature_writer.Write(feature_reader.Key(), nnet_out_host);

      // progress log,
      if (num_done % 100 == 0) {
        time_now = time.Elapsed();
        KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
                      << time_now/60 << " min; processed " << tot_t/time_now
                      << " frames per second.";
      }
      num_done++;
      tot_t += e_mat.NumRows();
    }

    // final message,
    KALDI_LOG << "Done " << num_done << " files"
              << " in " << time.Elapsed()/60 << "min,"
              << " (fps " << tot_t/time.Elapsed() << ")";

#if HAVE_CUDA == 1
    if (kaldi::g_kaldi_verbose_level >= 1) {
      CuDevice::Instantiate().PrintProfile();
    }
#endif

    if (num_done == 0) return -1;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
