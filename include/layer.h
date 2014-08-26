/*
 * Copyright 2014 by Institute of Acoustics, Chinese Academy of Sciences
 * http://www.ioa.ac.cn
 *
 * See the file COPYING for the licence associated with this software.
 *
 * Author(s):
 *   Xingyu Na, August 2014
 */

#ifndef LAYER_H_
#define LAYER_H_

#include "common.h"

using namespace std;

namespace xnn {

    class layer {
    public:
        typedef Tensor4d Tensor;
        enum elayertype {
            eFullLayer,
            eConvLayer,
            ePoolLayer,
            eDataLayer,
            eSoftLayer,
            eUnLayer
        } _ltype;

        /**** getters and setters ****/
        virtual const Tensor& get_a() const = 0;
        virtual Tensor& get_w() { return weights_; };
        virtual Tensor& get_b() { return biases_; };
        virtual Tensor& get_deda() { return de_da_; };
        virtual Tensor& get_dedw() { return de_dw_; };
        virtual Tensor& get_dedb() { return de_db_; };
        virtual bool is_trainable() const { return is_trainable_; };
        virtual UINT get_nneu() const = 0;
        virtual UINT get_nneu_lm1() const = 0;
        virtual string get_name() const { return name_; };
        virtual elayertype get_ltype() = 0;
        virtual void set_a(const Tensor& data) = 0;

        /**** utilities ****/
        virtual void initial() = 0;
        virtual void propagate(const Tensor& input) = 0;
        /* for those the (l - 1) layer terminate the error back propagation */
        virtual void backprop(const Tensor& input) = 0;
        /* for those the (l - 1) layer need to back propagate error */
        virtual void backprop(const Tensor& input, Tensor& deda_lm1) = 0;
        /* update weights and biases for weighted layer */
        virtual void updateweights() = 0;

    protected:
        virtual void reset() = 0;
        /* variables */
        Tensor a_;               /* neuron activations */
        Tensor weights_;         /* weight */
        Tensor biases_;          /* bias */
        Tensor de_dw_;           /* weight gradient, #input x #neurons
                                    used as accumulator in parallel training */
        Tensor de_db_;           /* bias gradient, #neurons x 1
                                    used as accumulator in parallel training */
        Tensor de_da_;           /* error of current layer */
        Tensor delta_;           /* error w.r.t. the to-be-activated variable */
        bool is_trainable_;
        string name_;
    };
}

#endif
