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
#include "layer.h"

namespace xnn {

    class clayer: public layer {
    public:
        clayer();
        clayer(const UINT nsamples, const UINT nfilters, const UINT filter_height, const UINT filter_width, 
               const UINT nchannels, const UINT image_height, const UINT image_width, const std::string name);
        virtual ~clayer();

        /* utilities */
        void initial();
        void propagate(const Tensor4d& input);
        void backprop(const Tensor4d& input);
        void backprop(const Tensor4d& input, Tensor4d& deda_lm1);
        void computegrad(const Tensor4d & input) {};
        void updateweights();

        /* getters and setters */
        void set_a (const Tensor4d& data);
        const Tensor4d & get_a () const { return a_; };
		UINT get_nneu() const { return 0; };
        UINT get_nneu_lm1() const { return 0; };
//        typename layer::elayertype get_ltype() { return this->eConvLayer; };
		elayertype get_ltype() { return this->eConvLayer; };
    protected:
        void reset();
        void active_function_ (const Tensor4d & sum);
        void der_active_function_ ();
        float activate (float input) { return input; };
        float der_activate (float input) { return 1.0; };
        UINT nsamples_;
        UINT nfilters_;
        UINT nchannels_;
        UINT filter_height_;
        UINT filter_width_;
        float learning_rate_;
    };
}

#endif
