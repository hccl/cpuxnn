/*
 * Copyright 2014 by Institute of Acoustics, Chinese Academy of Sciences
 * http://www.ioa.ac.cn
 *
 * See the file COPYING for the licence associated with this software.
 *
 * Author(s):
 *   Xingyu Na, June 2014
 */

#ifndef _PLAYER_H
#define _PLAYER_H

#include "tensor.h"
#include "layer.h"
#include "neuron.h"

namespace xnn {

    class player: public layer {
    public:
        player();
        player(const UINT nsamples, const UINT nchannels, const UINT image_height, const UINT image_width,
               const UINT sampler_height, const UINT sampler_width, const string name, const char ptype, enumFuncType actType);
        ~player();

        /* utilities */
        void initial();
        void propagate(const Tensor4d& input);
        void backprop(const Tensor4d& input) {};
        void backprop(const Tensor4d& input, Tensor4d& deda_lm1);
        void updateweights() {};

        /* getters and setters */
        void set_a(const Tensor4d & data) {};
        const Tensor4d & get_a() const { return a_; };
		UINT get_nneu() const { return 0; };
        UINT get_nneu_lm1() const { return 0; };
//        typename layer::elayertype get_ltype() { return this->ePoolLayer; };
		elayertype get_ltype() { return this->ePoolLayer; };

    protected:
        void reset();
        void active_function_ (const Tensor4d & sum);
        void der_active_function_ ();
//        float activate (float input) { return input; };
//        float der_activate (float input) { return 1.0; };
        void propavg(const Tensor4d & input);
        void propmax(const Tensor4d & input);
        void backpropavg(const Tensor4d & input, Tensor4d & deda_lm1);
        void backpropmax(const Tensor4d & input, Tensor4d & deda_lm1);

        enum epooltype {
            eAvgPool,
            eMaxPool,
            eUnPool,
        } ptype_;

        UINT nsamples_;
        UINT nchannels_;
        UINT sampler_height_;
        UINT sampler_width_;

		Neuron neurons_;
		enumFuncType funcType_;
    };
}

#endif
