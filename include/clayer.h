/*
 * Copyright 2014 by Institute of Acoustics, Chinese Academy of Sciences
 * http://www.ioa.ac.cn
 *
 * See the file COPYING for the licence associated with this software.
 *
 * Author(s):
 *   Xingyu Na, June 2014
 */

#ifndef _CLAYER_H
#define _CLAYER_H

#include "tensor.h"

namespace xnn {

    class clayer {
    public:
        clayer();
        clayer(const UINT nsamples, const UINT nfilters, const UINT filter_height, const UINT filter_width, 
               const UINT nchannels, const UINT image_height, const UINT image_width, const std::string name);
        virtual ~clayer();

        /* utilities */
        virtual void initial();
        virtual void propagate(const Tensor4d & input);
        virtual void backprop(const Tensor4d & input, Tensor4d & deda_lm1);
        virtual void computegrad(const Tensor4d & input) {};
        virtual void updateweights();

        /* getters and setters */
        void set_a (const Tensor4d& data);
        Tensor4d & get_a () { return a_; };
        Tensor4d & get_w () { return weights_; };
        Tensor4d & get_b () { return biases_; };
        Tensor4d & get_deda () { return de_da_; };
        Tensor4d & get_dedw () { return de_dw_; };
        Tensor4d & get_dedb () { return de_db_; };

    protected:
        void reset();
        void active_function_ (const Tensor4d & sum);
        void der_active_function_ ();
        float activate (float input) { return input; };
        float der_activate (float input) { return 1.0; };
        Tensor4d a_;
        Tensor4d weights_;
        Tensor4d biases_;
        Tensor4d de_dw_;
        Tensor4d de_db_;
        Tensor4d de_da_;
        Tensor4d delta_;
        UINT nsamples_;
        UINT nfilters_;
        UINT nchannels_;
        UINT filter_height_;
        UINT filter_width_;
        float learning_rate_;
        std::string name_;
    };
}

#endif
