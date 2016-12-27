// nnet/nnet-loss.cc

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

#include <sstream>
#include <iterator>
#include <algorithm>

#include "nnet/nnet-loss.h"
#include "nnet/nnet-utils.h"
#include "cudamatrix/cu-math.h"
#include "hmm/posterior.h"

namespace kaldi {
namespace nnet1 {


/* Xent */

/**
 * Helper function of Xent::Eval,
 * calculates number of matching elemente in 'hyp', 'ref' weighted by 'weights'.
 */
template <typename T>



inline void CountCorrectFramesWeighted(const CuArray<T> &hyp,
                                       const CuArray<T> &ref,
                                       const CuVectorBase<BaseFloat> &weights,
                                       Vector<double> *correct) {
  KALDI_ASSERT(hyp.Dim() == ref.Dim());
  KALDI_ASSERT(hyp.Dim() == weights.Dim());
  int32 dim = hyp.Dim();
  // Get GPU data to host,
  std::vector<T> hyp_h(dim), ref_h(dim);
  hyp.CopyToVec(&hyp_h);
  ref.CopyToVec(&ref_h);
  Vector<BaseFloat> w(dim);
  weights.CopyToVec(&w);
  // Accumulate weighted counts of correct frames,
  for (int32 i = 0; i < dim; i++) {
    KALDI_ASSERT(ref_h[i] < correct->Dim());
    (*correct)(ref_h[i]) += w(i) * (hyp_h[i] == ref_h[i] ? 1.0 : 0.0);
  }
} 

// <shr> added by 2016.7.8
/*inline void CalCosineFun(const CuMatrixBase<BaseFloat> &en_net_out,
                 const CuMatrixBase<BaseFloat> &fr_net_out,
                 CuVector<BaseFloat> *cos_value
                        ) {
  CuMatrix<BaseFloat> cal_tmp(en_net_out); 
  cal_tmp.MulElements(en_net_out);
  cal_tmp.MulElements(fr_net_out);
  cal_tmp.MulElements(fr_net_out);
  CuVector<BaseFloat> cos_value_tmp(cal_tmp.NumRows(),kSetZero);
  cos_value_tmp.AddRowSumMat(1.0, cal_tmp);
  cos_value->CopyFromVec(cos_value_tmp);

}*/

void Xent::EvalCosineLoss(const CuMatrixBase<BaseFloat> &en_net_out,
                const CuMatrixBase<BaseFloat> &fr_net_out,
                const CuVector<BaseFloat> &targets,
                CuMatrix<BaseFloat> *en_diff,
                CuMatrix<BaseFloat> *fr_diff) {
  // check inputs,
  KALDI_ASSERT(en_net_out.NumRows() == targets.Dim());
  KALDI_ASSERT(en_net_out.NumRows() == fr_net_out.NumRows());
  KALDI_ASSERT(en_net_out.NumCols() == fr_net_out.NumCols());
  KALDI_ASSERT(KALDI_ISFINITE(en_net_out.Sum()));
  KALDI_ASSERT(KALDI_ISFINITE(fr_net_out.Sum()));
  KALDI_ASSERT(KALDI_ISFINITE(targets.Sum()));

  int32 seq_num = en_net_out.NumRows();
  // buffer initialization,
  if (frames_.Dim() == 0) {
    frames_.Resize(seq_num);
    //loss_cos_.Resize(seq_num);
    correct_.Resize(seq_num); 
    correct_aux_.Resize(seq_num);
  }
  
  // copy data to GPU
  CuMatrix<BaseFloat> EME(en_net_out);
  CuMatrix<BaseFloat> EMF(en_net_out);
  CuMatrix<BaseFloat> FMF(fr_net_out);
  EME.MulElements(en_net_out); // E .* E
  EMF.MulElements(fr_net_out); // E .* F
  FMF.MulElements(fr_net_out); // F .* F
  CuVector<BaseFloat> ee(seq_num, kSetZero);
  CuVector<BaseFloat> ef(seq_num, kSetZero);
  CuVector<BaseFloat> ff(seq_num, kSetZero);
  //KALDI_LOG << "EME's size is: " << EME.NumRows() << ", " << EME.NumCols();
  ee.AddColSumMat(1.0, EME, 0.0); // sum(E .* E, 2)
  ef.AddColSumMat(1.0, EMF, 0.0); // sum(E .* F, 2)
  ff.AddColSumMat(1.0, FMF, 0.0); // sum(F .* F, 2)
  ee.Add(1e-20); ff.Add(1e-20); // avoid 0 in div operate
  CuVector<BaseFloat> iv_eeff(ee);
  iv_eeff.MulElements(ff); iv_eeff.Add(1e-20); iv_eeff.InvertElements(); // 1 ./ ee ./ ff
  CuVector<BaseFloat> tmp(ef); 
  tmp.MulElements(iv_eeff); tmp.Scale(2.0); // tmp = 2*ef./eeff 

  CuVector<BaseFloat> F(ef); 
  F.MulElements(tmp); F.Scale(0.5); // F = 0.5 * tmp .* ef
  
  CuVector<BaseFloat> f_aux(F); // TODO

  CuVector<BaseFloat> TMP(F);
  TMP.AddVec(-1.0, targets); 
  CuVector<BaseFloat> loss_cos_aux_(TMP); loss_cos_aux_.MulElements(TMP); // TODO
  TMP.MulElements(tmp); TMP.Scale(2.0); // TMP = 2 * (F-t) .* tmp
  
  // du,dv
  CuVector<BaseFloat> alpha(ee);
  CuVector<BaseFloat> beta(ff);
  alpha.InvertElements(); alpha.MulElements(ef); alpha.Scale(-1.0); // alpha = - ef ./ ee
  beta.InvertElements(); beta.MulElements(ef); beta.Scale(-1.0); // beta = - ef ./ ff
  *en_diff = en_net_out; en_diff->MulRowsVec(alpha); en_diff->AddMat(1.0, fr_net_out); en_diff->MulRowsVec(TMP);
  *fr_diff = fr_net_out; fr_diff->MulRowsVec(beta); fr_diff->AddMat(1.0, en_net_out); fr_diff->MulRowsVec(TMP);

  // calculate the acc and cos_2_fun
  float tmp_judge;
  for (int32 i = 0; i < seq_num; i++)
  {
    tmp_judge = (f_aux(i) >= 0.5) ? 1: 0;
    frames_(i) += 1;
    if (tmp_judge == targets(i)) {
      correct_(i) += 1;
      correct_aux_(i) += 1;
    }
  }
  /*loss_cos_aux_.Add(1e-20);
  loss_cos_aux_.ApplyLog();
  loss_cos_ += -loss_cos_aux_.Sum();*/
  loss_cos_ += loss_cos_aux_.Sum();
  //loss_cos_.AddVec(-1.0, loss_cos_aux_);
  // progressive loss reporting
  {
    static const int32 progress_step = 10000;  // 1h
    frames_progress_ += seq_num;
    xentropy_progress_ += loss_cos_aux_.Sum();

    KALDI_ASSERT(KALDI_ISFINITE(xentropy_progress_));

    if (frames_progress_ >= progress_step) {
      double progress_value =
        xentropy_progress_ / frames_progress_;    // -2log(f-target)
      // print,
      KALDI_VLOG(1) << "ProgressLoss[last "
                    << static_cast<int>(frames_progress_/10000) << "h of "
                    << static_cast<int>(frames_.Sum()/10000) << "h]: "
                    << progress_value << " (per-seq loss), "
                    << "Acc is: " << 100.0 * correct_aux_.Sum() / frames_progress_;   // TODO
      // store,
      loss_vec_.push_back(progress_value);
      // reset,
      frames_progress_ = 0;
      xentropy_progress_ = 0.0;
      correct_aux_.Resize(seq_num, kSetZero);
    }
  }
}

std::string Xent::CosLossReport() {
  double loss_value =
    loss_cos_ / frames_.Sum();
  std::ostringstream oss;
  oss << "AvgLoss: " << loss_value << " (-2log|f-target|)"
      << std::endl;

  oss << "progress: [";
  std::copy(loss_vec_.begin(), loss_vec_.end(),
            std::ostream_iterator<float>(oss, " "));
  oss << "]" << std::endl;

  double frame_accuracy = 100.0 * correct_.Sum() / frames_.Sum();
  oss << "FRAME_ACCURACY >> " << frame_accuracy << "% <<" << std::endl;

  return oss.str();
}

// </shr>




void Xent::Eval(const VectorBase<BaseFloat> &frame_weights,
                const CuMatrixBase<BaseFloat> &net_out,
                const CuMatrixBase<BaseFloat> &targets,
                CuMatrix<BaseFloat> *diff) {
  // check inputs,
  KALDI_ASSERT(net_out.NumCols() == targets.NumCols());
  KALDI_ASSERT(net_out.NumRows() == targets.NumRows());
  KALDI_ASSERT(net_out.NumRows() == frame_weights.Dim());

  KALDI_ASSERT(KALDI_ISFINITE(frame_weights.Sum()));
  KALDI_ASSERT(KALDI_ISFINITE(net_out.Sum()));
  KALDI_ASSERT(KALDI_ISFINITE(targets.Sum()));

  // buffer initialization,
  int32 num_classes = targets.NumCols();
  if (frames_.Dim() == 0) {
    frames_.Resize(num_classes);
    xentropy_.Resize(num_classes);
    entropy_.Resize(num_classes);
    correct_.Resize(num_classes);
  }

  // get frame_weights to GPU,
  frame_weights_ = frame_weights;

  // There may be frames for which the sum of targets is zero.
  // This happens in multi-lingual training when the frame
  // has target class in the softmax of another language.
  // We 'switch-off' such frames by masking the 'frame_weights_',
  target_sum_.Resize(targets.NumRows());
  target_sum_.AddColSumMat(1.0, targets, 0.0);
  frame_weights_.MulElements(target_sum_);

  // compute derivative wrt. activations of last layer of neurons,
  *diff = net_out;
  diff->AddMat(-1.0, targets);
  diff->MulRowsVec(frame_weights_);  // weighting,

  // count frames per class,
  frames_aux_ = targets;
  frames_aux_.MulRowsVec(frame_weights_);
  frames_.AddRowSumMat(1.0, CuMatrix<double>(frames_aux_));

  // evaluate the frame-level classification,
  net_out.FindRowMaxId(&max_id_out_);  // find max in nn-output
  targets.FindRowMaxId(&max_id_tgt_);  // find max in targets
  CountCorrectFramesWeighted(max_id_out_, max_id_tgt_,
                             frame_weights_, &correct_);

  // calculate cross_entropy (in GPU),
  xentropy_aux_ = net_out;  // y
  xentropy_aux_.Add(1e-20);  // avoid log(0)
  xentropy_aux_.ApplyLog();  // log(y)
  xentropy_aux_.MulElements(targets);  // t*log(y)
  xentropy_aux_.MulRowsVec(frame_weights_);  // w*t*log(y)
  xentropy_.AddRowSumMat(-1.0, CuMatrix<double>(xentropy_aux_));

  // caluculate entropy (in GPU),
  entropy_aux_ = targets;  // t
  entropy_aux_.Add(1e-20);  // avoid log(0)
  entropy_aux_.ApplyLog();  // log(t)
  entropy_aux_.MulElements(targets);  // t*log(t)
  entropy_aux_.MulRowsVec(frame_weights_);  // w*t*log(t)
  entropy_.AddRowSumMat(-1.0, CuMatrix<double>(entropy_aux_));

  // progressive loss reporting
  {
    //static const int32 progress_step = 3600*100;  // 1h
    static const int32 progress_step = 10000;  // 1h
    frames_progress_ += frame_weights_.Sum();
    xentropy_progress_ += -xentropy_aux_.Sum();
    entropy_progress_ += -entropy_aux_.Sum();

    //KALDI_ASSERT(KALDI_ISFINITE(xentropy_progress_));    // commented by shr
    //KALDI_ASSERT(KALDI_ISFINITE(entropy_progress_));  // commented by shr

    if (frames_progress_ > progress_step) {
      double progress_value =
        (xentropy_progress_ - entropy_progress_) / frames_progress_;
      /*// print,
      KALDI_VLOG(1) << "ProgressLoss[last "
                    << static_cast<int>(frames_progress_/100/3600) << "h of "
                    << static_cast<int>(frames_.Sum()/100/3600) << "h]: "
                    << progress_value << " (Xent)";*/
      // print,
      double acc = 100.0 * correct_.Sum() / frames_.Sum();
      KALDI_VLOG(1) << "ProgressLoss[last "
                    << static_cast<int>(frames_progress_/10000) << "h of "
                    << static_cast<int>(frames_.Sum()/10000) << "h]: "
                    << "Acc is: " << acc << ", "
                    << progress_value << " (Xent)";
      // store,
      loss_vec_.push_back(progress_value);
      // reset,
      frames_progress_ = 0;
      xentropy_progress_ = 0.0;
      entropy_progress_ = 0.0;
    }
  }
}


void Xent::Eval(const VectorBase<BaseFloat> &frame_weights,
                const CuMatrixBase<BaseFloat> &net_out,
                const Posterior &post,
                CuMatrix<BaseFloat> *diff) {
  int32 num_frames = net_out.NumRows(),
    num_pdf = net_out.NumCols();
  KALDI_ASSERT(num_frames == post.size());

  // convert posterior to matrix,
  PosteriorToMatrix(post, num_pdf, &tgt_mat_);

  // call the other eval function,
  Eval(frame_weights, net_out, tgt_mat_, diff);
}


std::string Xent::Report() {
  double loss_value =
    (xentropy_.Sum() - entropy_.Sum()) / frames_.Sum();
  std::ostringstream oss;
  oss << "AvgLoss: " << loss_value << " (Xent), "
      << "[AvgXent: " << xentropy_.Sum() / frames_.Sum()
      << ", AvgTargetEnt: " << entropy_.Sum() / frames_.Sum()
      << "]" << std::endl;

  oss << "progress: [";
  std::copy(loss_vec_.begin(), loss_vec_.end(),
            std::ostream_iterator<float>(oss, " "));
  oss << "]" << std::endl;

  double frame_accuracy = 100.0 * correct_.Sum() / frames_.Sum();
  oss << "FRAME_ACCURACY >> " << frame_accuracy << "% <<" << std::endl;

  return oss.str();
}


std::string Xent::ReportPerClass() {
  std::ostringstream oss;
  oss << "PER-CLASS PERFORMANCE:" << std::endl;
  oss << "@@@ Frames per-class:" << frames_;
  // get inverted counts,
  CuVector<double> inv_frames(frames_);
  inv_frames.ApplyPow(-1.0);
  // loss, kl = xentropy-entropy,
  CuVector<double> loss(xentropy_);
  loss.AddVec(-1.0, entropy_);
  loss.MulElements(inv_frames);
  oss << "@@@ Loss per-class:" << loss;
  // frame accuracy (assuming targets are binary),
  CuVector<double> frm_accu(correct_);
  frm_accu.MulElements(inv_frames);
  frm_accu.Scale(100.0);
  oss << "@@@ Frame-accuracy per-class:" << frm_accu;
  //
  return oss.str();
}


/* Mse */

void Mse::Eval(const VectorBase<BaseFloat> &frame_weights,
               const CuMatrixBase<BaseFloat>& net_out,
               const CuMatrixBase<BaseFloat>& target,
               CuMatrix<BaseFloat>* diff) {
  // check inputs,
  KALDI_ASSERT(net_out.NumCols() == target.NumCols());
  KALDI_ASSERT(net_out.NumRows() == target.NumRows());
  KALDI_ASSERT(net_out.NumRows() == frame_weights.Dim());

  KALDI_ASSERT(KALDI_ISFINITE(frame_weights.Sum()));
  KALDI_ASSERT(KALDI_ISFINITE(net_out.Sum()));
  KALDI_ASSERT(KALDI_ISFINITE(target.Sum()));

  int32 num_frames = frame_weights.Sum();
  KALDI_ASSERT(num_frames >= 0.0);

  // get frame_weights to GPU,
  frame_weights_ = frame_weights;

  // compute derivative w.r.t. neural nerwork outputs
  *diff = net_out;  // y
  diff->AddMat(-1.0, target);  // (y - t)
  diff->MulRowsVec(frame_weights_);  // weighting,

  // Compute MeanSquareError loss of mini-batch
  diff_pow_2_ = *diff;
  diff_pow_2_.MulElements(diff_pow_2_);  // (y - t)^2
  diff_pow_2_.MulRowsVec(frame_weights_);  // w*(y - t)^2
  double mean_square_error = 0.5 * diff_pow_2_.Sum();  // sum the matrix,

  KALDI_ASSERT(KALDI_ISFINITE(mean_square_error));

  // accumulate
  loss_ += mean_square_error;
  frames_ += num_frames;

  // progressive loss reporting
  {
    static const int32 progress_step = 3600*100;  // 1h
    frames_progress_ += num_frames;
    loss_progress_ += mean_square_error;
    if (frames_progress_ > progress_step) {
      KALDI_VLOG(1) << "ProgressLoss[last "
                    << static_cast<int>(frames_progress_/100/3600) << "h of "
                    << static_cast<int>(frames_/100/3600) << "h]: "
                    << loss_progress_/frames_progress_ << " (Mse)";
      // store
      loss_vec_.push_back(loss_progress_/frames_progress_);
      // reset
      frames_progress_ = 0;
      loss_progress_ = 0.0;
    }
  }
}


void Mse::Eval(const VectorBase<BaseFloat> &frame_weights,
               const CuMatrixBase<BaseFloat>& net_out,
               const Posterior& post,
               CuMatrix<BaseFloat>* diff) {
  int32 num_frames = net_out.NumRows(),
    num_nn_outputs = net_out.NumCols();
  KALDI_ASSERT(num_frames == post.size());

  // convert posterior to matrix,
  PosteriorToMatrix(post, num_nn_outputs, &tgt_mat_);

  // call the other eval function,
  Eval(frame_weights, net_out, tgt_mat_, diff);
}


std::string Mse::Report() {
  // compute root mean square,
  int32 num_tgt = diff_pow_2_.NumCols();
  BaseFloat root_mean_square = sqrt(loss_/frames_/num_tgt);
  // build the message,
  std::ostringstream oss;
  oss << "AvgLoss: " << loss_/frames_ << " (Mse), "
      << "[RMS " << root_mean_square << ", frames "
      << frames_ << "]" << std::endl;
  oss << "progress: [";
  std::copy(loss_vec_.begin(), loss_vec_.end(),
            std::ostream_iterator<float>(oss, " "));
  oss << "]" << std::endl;
  return oss.str();
}


/* MultiTaskLoss */

void MultiTaskLoss::InitFromString(const std::string& s) {
  std::vector<std::string> v;
  SplitStringToVector(s, ",:" /* delimiter */, false, &v);

  KALDI_ASSERT((v.size()-1) % 3 == 0);  // triplets,
  KALDI_ASSERT(v[0] == "multitask");  // header,

  // parse the definition of multitask loss,
  std::vector<std::string>::iterator it(v.begin()+1);  // skip header,
  for ( ; it != v.end(); ++it) {
    // type,
    if (*it == "xent") {
      loss_vec_.push_back(new Xent());
    } else if (*it == "mse") {
      loss_vec_.push_back(new Mse());
    } else {
      KALDI_ERR << "Unknown objective function code : " << *it;
    }
    ++it;
    // dim,
    int32 dim;
    if (!ConvertStringToInteger(*it, &dim)) {
      KALDI_ERR << "Cannot convert 'dim' " << *it << " to integer!";
    }
    loss_dim_.push_back(dim);
    ++it;
    // weight,
    BaseFloat weight;
    if (!ConvertStringToReal(*it, &weight)) {
      KALDI_ERR << "Cannot convert 'weight' " << *it << " to integer!";
    }
    KALDI_ASSERT(weight >= 0.0);
    loss_weights_.push_back(weight);
  }

  // build vector with starting-point offsets,
  loss_dim_offset_.resize(loss_dim_.size()+1, 0);  // 1st zero stays,
  for (int32 i = 1; i <= loss_dim_.size(); i++) {
    loss_dim_offset_[i] = loss_dim_offset_[i-1] + loss_dim_[i-1];
  }

  // sanity check,
  KALDI_ASSERT(loss_vec_.size() > 0);
  KALDI_ASSERT(loss_vec_.size() == loss_dim_.size());
  KALDI_ASSERT(loss_vec_.size() == loss_weights_.size());
}

void MultiTaskLoss::Eval(const VectorBase<BaseFloat> &frame_weights,
            const CuMatrixBase<BaseFloat>& net_out,
            const Posterior& post,
            CuMatrix<BaseFloat>* diff) {
  int32 num_frames = net_out.NumRows(),
    num_output = net_out.NumCols();
  KALDI_ASSERT(num_frames == post.size());
  KALDI_ASSERT(num_output == loss_dim_offset_.back());  // sum of loss-dims,

  // convert posterior to matrix,
  PosteriorToMatrix(post, num_output, &tgt_mat_);

  // allocate diff matrix,
  diff->Resize(num_frames, num_output);

  // call the vector of loss functions,
  CuMatrix<BaseFloat> diff_aux;
  for (int32 i = 0; i < loss_vec_.size(); i++) {
    loss_vec_[i]->Eval(frame_weights,
      net_out.ColRange(loss_dim_offset_[i], loss_dim_[i]),
      tgt_mat_.ColRange(loss_dim_offset_[i], loss_dim_[i]),
      &diff_aux);
    // Scale the gradients,
    diff_aux.Scale(loss_weights_[i]);
    // Copy to diff,
    diff->ColRange(loss_dim_offset_[i], loss_dim_[i]).CopyFromMat(diff_aux);
  }
}

std::string MultiTaskLoss::Report() {
  // calculate overall loss (weighted),
  BaseFloat overall_loss = AvgLoss();
  // copy the loss-values into a vector,
  std::vector<BaseFloat> loss_values;
  for (int32 i = 0; i < loss_vec_.size(); i++) {
    loss_values.push_back(loss_vec_[i]->AvgLoss());
  }

  // build the message,
  std::ostringstream oss;
  oss << "MultiTaskLoss, with " << loss_vec_.size()
      << " parallel loss functions." << std::endl;
  // individual loss reports first,
  for (int32 i = 0; i < loss_vec_.size(); i++) {
    oss << "Loss " << i+1 << ", " << loss_vec_[i]->Report() << std::endl;
  }

  // overall loss is last,
  oss << "Loss (OVERALL), "
      << "AvgLoss: " << overall_loss << " (MultiTaskLoss), "
      << "weights " << loss_weights_ << ", "
      << "values " << loss_values << std::endl;

  return oss.str();
}

BaseFloat MultiTaskLoss::AvgLoss() {
  BaseFloat ans(0.0);
  for (int32 i = 0; i < loss_vec_.size(); i++) {
    BaseFloat val = loss_weights_[i] * loss_vec_[i]->AvgLoss();
    if (!KALDI_ISFINITE(val)) {
      KALDI_WARN << "Loss " << i+1 << ", has bad objective function value '"
                 << val << "', using 0.0 instead.";
      val = 0.0;
    }
    ans += val;
  }
  return ans;
}

}  // namespace nnet1
}  // namespace kaldi
