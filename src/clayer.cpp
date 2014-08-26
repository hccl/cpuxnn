/*
 * Copyright 2014 by Institute of Acoustics, Chinese Academy of Sciences
 * http://www.ioa.ac.cn
 *
 * See the file COPYING for the licence associated with this software.
 *
 * Author(s):
 *   Xingyu Na, June 2014
 */

#include "clayer.h"

namespace xnn {

    clayer::clayer() {
        reset();
    }

    clayer::~clayer() {
        reset();
    }

    void clayer::reset() {
        nsamples_ = 0;
        nfilters_ = 0;
        nchannels_ = 0;
        filter_height_ = 0;
        filter_width_ = 0;
    }

    clayer::clayer(const UINT nsamples, const UINT nfilters, const UINT filter_height, const UINT filter_width, 
                   const UINT nchannels, const UINT image_height, const UINT image_width, const std::string name) {
        nsamples_ = nsamples;
        nfilters_ = nfilters;
        nchannels_ = nchannels;
        filter_height_ = filter_height;
        filter_width_ = filter_width;
        name_ = name;
        a_ = Tensor4d(nsamples, nfilters, image_height - filter_height + 1, image_width - filter_width + 1);
        de_da_ = Tensor4d(nsamples, nfilters, image_height - filter_height + 1, image_width - filter_width + 1);
        delta_ = Tensor4d(nsamples, nfilters, image_height - filter_height + 1, image_width - filter_width + 1);
        weights_ = Tensor4d(nchannels, nfilters, filter_height, filter_width);
        de_dw_ = Tensor4d(nchannels, nfilters, filter_height, filter_width);
        biases_ = Tensor4d(1, nfilters, 1, 1);
        de_db_ = Tensor4d(1, nfilters, 1, 1);
    }

    void clayer::initial() {
        weights_.rand_elt();
        biases_.rand_elt();
        de_dw_.zeros_elt();
        de_db_.zeros_elt();
        de_da_.zeros_elt();
        delta_.zeros_elt();
        learning_rate_ = 0.001;
    }

    void clayer::propagate(const Tensor4d & input) {
        /* 4d tensor storing output feature maps */
        Tensor4d *output = new Tensor4d(input.get_dim(0), weights_.get_dim(1), 
                                        input.get_dim(2) - weights_.get_dim(2) + 1,
                                        input.get_dim(3) - weights_.get_dim(3) + 1); 
        float bias;

        output->tnrProductValid(input, weights_);
        for(UINT i = 0; i < nsamples_; ++i) {
            for(UINT j = 0; j < nfilters_; ++j) {
                bias = biases_.get_elt(0, j, 0, 0);
                output->get_kernel(i, j).slrAdd(bias);
            }
        }
        active_function_(*output);
        delete output; output = NULL;
    }

    void clayer::backprop(const Tensor4d & input) {
        Tensor4d * transp = new Tensor4d(input.get_dim(1), input.get_dim(0), input.get_dim(2), input.get_dim(3));

        der_active_function_();
        delta_.eltProduct(this->de_da_);
        transp->transpose(input);
        de_dw_.tnrProductValid(*transp, delta_);
        de_db_.tnrSum(delta_, 1);

        delete transp; transp = NULL;
    }

    void clayer::backprop(const Tensor4d & input, Tensor4d & deda_lm1) {
        Tensor4d * transp = new Tensor4d(input.get_dim(1), input.get_dim(0), input.get_dim(2), input.get_dim(3));
        Tensor4d * wt = new Tensor4d(weights_.get_dim(1), weights_.get_dim(0), weights_.get_dim(2), weights_.get_dim(3));
        der_active_function_();
        delta_.eltProduct(this->de_da_);
        transp->transpose(input);
        de_dw_.tnrProductValid(*transp, delta_);
        de_db_.tnrSum(delta_, 1);
        wt->transpose(weights_);
        deda_lm1.tnrProductFull(delta_, *wt);

        delete wt; wt = NULL;
        delete transp; transp = NULL;
    }

    void clayer::updateweights() {
        de_dw_.slrProduct(learning_rate_);
        de_db_.slrProduct(learning_rate_);
        weights_.tnrAdd(de_dw_);
        biases_.tnrAdd(de_db_);
    }

    void clayer::active_function_ ( const Tensor4d & sum ) {
        for(UINT i = 0; i < sum.get_dim(0); ++i) {
            for(UINT j = 0; j < sum.get_dim(1); ++j) {
                for(UINT m = 0; m < sum.get_dim(2); ++m)
                    for(UINT n = 0; n < sum.get_dim(3); ++n)
                        a_.set_elt(i, j, m, n, activate(sum.get_elt(i, j, m, n)));
            }
        }
    }

    void clayer::der_active_function_ () {
        for(UINT i = 0; i < a_.get_dim(0); ++i) {
            for(UINT j = 0; j < a_.get_dim(1); ++j) {
                for(UINT m = 0; m < a_.get_dim(2); ++m)
                    for(UINT n = 0; n < a_.get_dim(3); ++n)
                        delta_.set_elt(i, j, m, n, der_activate(a_.get_elt(i, j, m, n)));
            }
        }
    }

    void clayer::set_a (const Tensor4d& data) {
        a_.copy(data);
    }
}
