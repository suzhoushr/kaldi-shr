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

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        feature_wspecifier = po.GetArg(3);

    // Select the GPU
#if HAVE_CUDA == 1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    Nnet nnet;
    nnet.Read(model_filename);

    // disable dropout,
    nnet.SetDropoutRetention(1.0);

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    BaseFloatMatrixWriter feature_writer(feature_wspecifier);

    CuMatrix<BaseFloat> feats, feats_transf, nnet_out;
    Matrix<BaseFloat> nnet_out_host;

    Timer time;
    int32 num_done = 0;

    // main loop,
    for (; !feature_reader.Done(); feature_reader.Next()) {
      // read
      Matrix<BaseFloat> mat = feature_reader.Value();
      std::string utt = feature_reader.Key();

      // push it to gpu,
      feats = mat;

      // fwd-pass, nnet,
      Timer time_model;
      nnet.Feedforward(feats, &nnet_out);
      KALDI_LOG << "model elapsed time = " << time_model.Elapsed() * 1000 << " ms";

      // frame-level labels, by selecting the label with the largest probability at each frame
      CuArray<int32> maxid(nnet_out.NumRows());
      nnet_out.FindRowMaxId(&maxid);

      int32 dim = maxid.Dim();
  
      std::vector<int32> data(dim);
      maxid.CopyToVec(&data);

      // remove the repetitions
      int32 i = 1, j = 1;
      while(j < dim) {
          if (data[j] != data[j-1]) {
              data[i] = data[j];
              i++;
          }
          j++;
      }

      // remove the blanks
      std::vector<int32> hyp_seq(0);
      for (int32 n = 0; n < i; n++) {
          if (data[n] != 0) {
              hyp_seq.push_back(data[n]);
          }
      }

      // write,
      Matrix<BaseFloat> nnet_out_host(1, hyp_seq.size()+1);
      nnet_out_host(0,0) = 1;
      if(hyp_seq.size() == 0) nnet_out_host(0,0) = 0;
      for (int32 i =0; i < hyp_seq.size(); i++)  nnet_out_host(0,i+1) = hyp_seq[i];
      feature_writer.Write(feature_reader.Key(), nnet_out_host);

      num_done++;

    }


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
