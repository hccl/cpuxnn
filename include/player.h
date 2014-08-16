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

using namespace std;

namespace xnn {

    class player {
    public:
        player();
        player(const UINT nsamples, const UINT nchannels, const UINT image_height, const UINT image_width,
               const UINT sampler_height, const UINT sampler_width, const string name, const char ptype);
        ~player();

        /* utilities */
        virtual void initial();
        virtual void propagate(const Tensor4d& input);
        virtual void backprop(const Tensor4d& input, Tensor4d& deda_lm1);

        /* getters and setters */
        Tensor4d & get_a () { return a_; };
        Tensor4d & get_deda () { return de_da_; };

    protected:
        void reset();
        void active_function_ (const Tensor4d & sum);
        void der_active_function_ ();
        float activate (float input) { return input; };
        float der_activate (float input) { return 1.0; };
        void propavg(const Tensor4d & input);
        void propmax(const Tensor4d & input);
        void backpropavg(const Tensor4d & input, Tensor4d & deda_lm1);
        void backpropmax(const Tensor4d & input, Tensor4d & deda_lm1);

        enum epooltype {
            eAvgPool,
            eMaxPool,
            eUnPool,
        } ptype_;
        Tensor4d a_;
        Tensor4d de_da_;
        Tensor4d delta_;
        UINT nsamples_;
        UINT nchannels_;
        UINT sampler_height_;
        UINT sampler_width_;
        string name_;
    };
}

#endif
