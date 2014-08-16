/*
 * Copyright 2014 by Institute of Acoustics, Chinese Academy of Sciences
 * http://www.ioa.ac.cn
 *
 * See the file COPYING for the licence associated with this software.
 *
 * Author(s):
 *   Xingyu Na, June 2014
 */

#include "player.h"

namespace xnn {

    player::player() {
        reset();
    }

    player::player(const UINT nsamples, const UINT nchannels, const UINT image_height, const UINT image_width,
                   const UINT sampler_height, const UINT sampler_width, const string name, const char ptype) {
        nsamples_ = nsamples;
        nchannels_ = nchannels;
        sampler_height_ = sampler_height;
        sampler_width_ = sampler_width;
        a_ = Tensor4d(nsamples, nchannels, image_height / sampler_height, image_width / sampler_width);
        de_da_ = Tensor4d(nsamples, nchannels, image_height / sampler_height, image_width / sampler_width);
        delta_ = Tensor4d(nsamples, nchannels, image_height / sampler_height, image_width / sampler_width);
        name_ = name;
        switch (ptype) {
        case 'm':
            ptype_ = player::eMaxPool;
            break;
        case 'a' :
            ptype_ = player::eAvgPool;
            break;
        default :
            ptype_ = player::eUnPool;
        }
    }

    player::~player() {
        reset();
    }

    void player::reset() {
        nsamples_ = 0;
        nchannels_ = 0;
        sampler_height_ = 0;
        sampler_width_ = 0;
    }

    void player::initial() {
        de_da_.zeros_elt();
    }
    
    void player::propagate(const Tensor4d & input) {
        assert(this->a_.get_dim(2) == input.get_dim(2) / this->sampler_height_ );
        assert(this->a_.get_dim(3) == input.get_dim(3) / this->sampler_width_ );

        switch(this->ptype_) {
        case player::eAvgPool:
            propavg(input);
            break;
        case player::eMaxPool:
            propmax(input);
            break;
        default:
            throw runtime_error("Unknown pooling type");
        }
    }

    void player::propmax(const Tensor4d & input) {
        Tensor4d *output = new Tensor4d(a_.get_dim(0), a_.get_dim(1), a_.get_dim(2), a_.get_dim(3));
        for(UINT ns = 0; ns < nsamples_; ++ns) {
            for(UINT nc = 0; nc < nchannels_; ++nc) {
                output->get_kernel(ns, nc).samplemax(input.get_kernel(ns, nc), sampler_height_, sampler_width_);
            }
        }
        active_function_(*output);
        delete output; output = NULL;
    }

    void player::propavg(const Tensor4d & input) {
        Tensor4d *output = new Tensor4d(a_.get_dim(0), a_.get_dim(1), a_.get_dim(2), a_.get_dim(3));
        for(UINT ns = 0; ns < nsamples_; ++ns) {
            for(UINT nc = 0; nc < nchannels_; ++nc) {
                output->get_kernel(ns, nc).sampleavg(input.get_kernel(ns, nc), sampler_height_, sampler_width_);
            }
        }
        active_function_(*output);
        delete output; output = NULL;
    }

    void player::backprop(const Tensor4d & input, Tensor4d & deda_lm1) {
        switch(this->ptype_) {
        case player::eAvgPool:
            backpropavg(input, deda_lm1);
            break;
        case player::eMaxPool:
            backpropmax(input, deda_lm1);
            break;
        default:
            throw runtime_error("Unknown pooling type");
        }
    }

    void player::backpropmax(const Tensor4d & input, Tensor4d & deda_lm1) {
        der_active_function_();
        for(UINT ns = 0; ns < nsamples_; ++ns)
            for(UINT nc = 0; nc < nchannels_; ++nc)
                deda_lm1.get_kernel(ns, nc).unsamplemax(delta_.get_kernel(ns, nc), input.get_kernel(ns, nc), sampler_height_, sampler_width_);
    }

    void player::backpropavg(const Tensor4d & input, Tensor4d & deda_lm1) {
        der_active_function_();
        for(UINT ns = 0; ns < nsamples_; ++ns)
            for(UINT nc = 0; nc < nchannels_; ++nc)
                deda_lm1.get_kernel(ns, nc).unsampleavg(delta_.get_kernel(ns, nc), sampler_height_, sampler_width_);
    }

    void player::active_function_ ( const Tensor4d & sum ) {
        for(UINT i = 0; i < sum.get_dim(0); ++i) {
            for(UINT j = 0; j < sum.get_dim(1); ++j) {
                for(UINT m = 0; m < sum.get_dim(2); ++m)
                    for(UINT n = 0; n < sum.get_dim(3); ++n)
                        a_.set_elt(i, j, m, n, activate(sum.get_elt(i, j, m, n)));
            }
        }
    }

    void player::der_active_function_ () {
        for(UINT i = 0; i < a_.get_dim(0); ++i)
            for(UINT j = 0; j < a_.get_dim(1); ++j)
                for(UINT m = 0; m < a_.get_dim(2); ++m)
                    for(UINT n = 0; n < a_.get_dim(3); ++n)
                        delta_.set_elt(i, j, m, n, der_activate(a_.get_elt(i, j, m, n)));
    }
}
