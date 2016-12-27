// nnet/nnet-loss.h

// Copyright 2011-2015  Brno University of Technology (author: Karel Vesely)

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

#ifndef KALDI_NNET_NNET_LOSS_H_
#define KALDI_NNET_NNET_LOSS_H_

#include <string>
#include <vector>

#include "base/kaldi-common.h"
#include "util/kaldi-holder.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"
#include "cudamatrix/cu-array.h"
#include "hmm/posterior.h"

namespace kaldi {
namespace nnet1 {


class LossItf {
 public:
  LossItf() { }
  virtual ~LossItf() { }

  /// Evaluate cross entropy using target-matrix (supports soft labels),
  virtual void Eval(const VectorBase<BaseFloat> &frame_weights,
            const CuMatrixBase<BaseFloat> &net_out,
            const CuMatrixBase<BaseFloat> &target,
            CuMatrix<BaseFloat> *diff) = 0;
  
  /*/// Evaluate cos-loss using target-matrix (supports soft labels), shr
  virtual void EvalCosineLoss(const CuMatrixBase<BaseFloat> &net1_out,   
            const CuMatrixBase<BaseFloat> &net2_out,
            const CuVector<BaseFloat> &target,
            CuMatrix<BaseFloat> *en_diff,
            CuMatrix<BaseFloat> *fr_diff) = 0; */

  /// Evaluate cross entropy using target-posteriors (supports soft labels),
  virtual void Eval(const VectorBase<BaseFloat> &frame_weights,
            const CuMatrixBase<BaseFloat> &net_out,
            const Posterior &target,
            CuMatrix<BaseFloat> *diff) = 0;

  /// Generate string with error report,
  virtual std::string Report() = 0;

  /// Get loss value (frame average),
  virtual BaseFloat AvgLoss() = 0;
};


class Xent : public LossItf {
 public:
  Xent():
    frames_progress_(0.0),
    xentropy_progress_(0.0),
    entropy_progress_(0.0),
    loss_cos_progress_(0.0),  // shr
    loss_cos_(0.0)  // shr
  { }

  ~Xent()
  { }

  /// Evaluate cross entropy using target-matrix (supports soft labels),
  void Eval(const VectorBase<BaseFloat> &frame_weights,
            const CuMatrixBase<BaseFloat> &net_out,
            const CuMatrixBase<BaseFloat> &target,
            CuMatrix<BaseFloat> *diff);

  /// Evaluate cross entropy using target-posteriors (supports soft labels),
  void Eval(const VectorBase<BaseFloat> &frame_weights,
            const CuMatrixBase<BaseFloat> &net_out,
            const Posterior &target,
            CuMatrix<BaseFloat> *diff);
  

  /// Evaluate cos-loss using target-matrix (supports soft labels), shr
  void EvalCosineLoss(const CuMatrixBase<BaseFloat> &net1_out,   
            const CuMatrixBase<BaseFloat> &net2_out,
            const CuVector<BaseFloat> &target,
            CuMatrix<BaseFloat> *en_diff,
            CuMatrix<BaseFloat> *fr_diff);

  /// Generate string with error report,
  std::string Report();

  /// Generate string with error report, shr
  std::string CosLossReport();

  /// Generate string with per-class error report,
  std::string ReportPerClass();

  /// Get loss value (frame average),
  BaseFloat AvgLoss() {
    return (xentropy_.Sum() - entropy_.Sum()) / frames_.Sum();
  }

 private:
  // main stats collected per target-class,
  CuVector<double> frames_;
  Vector<double> correct_;
  Vector<double> correct_aux_;
  CuVector<double> xentropy_;
  CuVector<double> entropy_;
  

  // partial results during training,
  double frames_progress_;
  double xentropy_progress_;
  double entropy_progress_;
  float loss_cos_progress_;
  float loss_cos_;
  std::vector<float> loss_vec_;

  // weigting buffer,
  CuVector<BaseFloat> frame_weights_;
  CuVector<BaseFloat> target_sum_;

  // loss computation buffers,
  CuMatrix<BaseFloat> tgt_mat_;
  CuMatrix<BaseFloat> frames_aux_;
  CuMatrix<BaseFloat> xentropy_aux_;
  CuMatrix<BaseFloat> entropy_aux_;

  // frame classification buffers,
  CuArray<int32> max_id_out_;
  CuArray<int32> max_id_tgt_;
};


class Mse : public LossItf {
 public:
  Mse():
    frames_(0.0),
    loss_(0.0),
    frames_progress_(0.0),
    loss_progress_(0.0)
  { }

  ~Mse()
  { }

  /// Evaluate mean square error using target-matrix,
  void Eval(const VectorBase<BaseFloat> &frame_weights,
            const CuMatrixBase<BaseFloat>& net_out,
            const CuMatrixBase<BaseFloat>& target,
            CuMatrix<BaseFloat>* diff);

  /// Evaluate mean square error using target-posteior,
  void Eval(const VectorBase<BaseFloat> &frame_weights,
            const CuMatrixBase<BaseFloat>& net_out,
            const Posterior& target,
            CuMatrix<BaseFloat>* diff);

  /// Generate string with error report
  std::string Report();

  /// Get loss value (frame average),
  BaseFloat AvgLoss() {
    return loss_ / frames_;
  }

 private:
  double frames_;
  double loss_;

  double frames_progress_;
  double loss_progress_;
  std::vector<float> loss_vec_;

  CuVector<BaseFloat> frame_weights_;
  CuMatrix<BaseFloat> tgt_mat_;
  CuMatrix<BaseFloat> diff_pow_2_;
};


class MultiTaskLoss : public LossItf {
 public:
  MultiTaskLoss()
  { }

  ~MultiTaskLoss() {
    while (loss_vec_.size() > 0) {
      delete loss_vec_.back();
      loss_vec_.pop_back();
    }
  }

  /// Initialize from string, the format for string 's' is :
  /// 'multitask,<type1>,<dim1>,<weight1>,...,<typeN>,<dimN>,<weightN>'
  ///
  /// Practically it can look like this :
  /// 'multitask,xent,2456,1.0,mse,440,0.001'
  void InitFromString(const std::string& s);

  /// Evaluate mean square error using target-matrix,
  void Eval(const VectorBase<BaseFloat> &frame_weights,
            const CuMatrixBase<BaseFloat>& net_out,
            const CuMatrixBase<BaseFloat>& target,
            CuMatrix<BaseFloat>* diff) {
    KALDI_ERR << "This is not supposed to be called!";
  }

  /// Evaluate mean square error using target-posteior,
  void Eval(const VectorBase<BaseFloat> &frame_weights,
            const CuMatrixBase<BaseFloat>& net_out,
            const Posterior& target,
            CuMatrix<BaseFloat>* diff);

  /// Generate string with error report
  std::string Report();

  /// Get loss value (frame average),
  BaseFloat AvgLoss();

 private:
  std::vector<LossItf*>  loss_vec_;
  std::vector<int32>     loss_dim_;
  std::vector<BaseFloat> loss_weights_;

  std::vector<int32>     loss_dim_offset_;

  CuMatrix<BaseFloat>    tgt_mat_;
};

// <ctc> added by shr, 2016.11.14
class Ctc {
 public:
  Ctc() : frames_(0), sequences_num_(0), ref_num_(0), error_num_(0), 
          frames_progress_(0), ref_num_progress_(0), error_num_progress_(0),
          sequences_progress_(0), obj_progress_(0.0), report_step_(100) { }
  ~Ctc() { }

  /// CTC training over a single sequence from the labels. The errors are returned to [diff]
  void Eval(const CuMatrixBase<BaseFloat> &net_out, const std::vector<int32> &label, CuMatrix<BaseFloat> *diff);

  /// CTC training over multiple sequences. The errors are returned to [diff]
  void EvalParallel(const std::vector<int32> &frame_num_utt, const CuMatrixBase<BaseFloat> &net_out,
                    std::vector< std::vector<int32> > &label, CuMatrix<BaseFloat> *diff);

  /// Compute token error rate from the softmax-layer activations and the given labels. From the softmax activations,
  /// we get the frame-level labels, by selecting the label with the largest probability at each frame. Then, the frame
  /// -level labels are shrunk by removing the blanks and collasping the repetitions. This gives us the utterance-level
  /// labels, from which we can compute the error rate. The error rate is the Levenshtein distance between the hyp labels
  /// and the given reference label sequence.
  void ErrorRate(const CuMatrixBase<BaseFloat> &net_out, const std::vector<int32> &label, float* err, std::vector<int32> *hyp);

  /// Compute token error rate over multiple sequences. 
  void ErrorRateMSeq(const std::vector<int> &frame_num_utt, const CuMatrixBase<BaseFloat> &net_out, std::vector< std::vector<int> > &label);

  /// Set the step of reporting
  void SetReportStep(int32 report_step) { report_step_ = report_step;  }

  /// Generate string with report
  std::string Report();

  float NumErrorTokens() const { return error_num_;}
  int32 NumRefTokens() const { return ref_num_;}

 private:
  int32 frames_;                    // total frame number
  int32 sequences_num_; 
  int32 ref_num_;                   // total number of tokens in label sequences
  float error_num_;                 // total number of errors (edit distance between hyp and ref)

  int32 frames_progress_;
  int32 ref_num_progress_;
  float error_num_progress_;

  int32 sequences_progress_;         // registry for the number of sequences
  double obj_progress_;              // registry for the optimization objective

  int32 report_step_;                // report obj and accuracy every so many sequences/utterances

  std::vector<int32> label_expand_;  // expanded version of the label sequence
  CuMatrix<BaseFloat> alpha_;        // alpha values
  CuMatrix<BaseFloat> beta_;         // beta values
  CuMatrix<BaseFloat> ctc_err_;      // ctc errors
};
// </ctc>


}  // namespace nnet1
}  // namespace kaldi

#endif  // KALDI_NNET_NNET_LOSS_H_

