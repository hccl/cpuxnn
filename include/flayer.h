/*
 * Copyright 2014 by Institute of Acoustics, Chinese Academy of Sciences
 * http://www.ioa.ac.cn
 *
 * See the file COPYING for the licence associated with this software.
 *
 * Author(s):
 *   Xingyu Na, August 2014
 */

#ifndef FLAYER_H_
#define FLAYER_H_

#include "tensor.h"
#include "layer.h"

using namespace std;

namespace xnn {

    /* for flayer, I use one cell of the tensor, a.k.a. a matrix to represent the variables */
    class flayer: public layer {
    public:
    	flayer();
        flayer(const UINT nsamples, const UINT nneurons, bool is_trainable, const std::string name);
        flayer(const UINT nsamples, const UINT nneurons, UINT nneurons_lm1, bool is_trainable, const std::string name);
        flayer(const Tensor4d& weights, const Tensor4d& biases, bool is_trainable, const std::string name);
        ~flayer();

        /* utilities */
        void initial();
        void propagate(const Tensor4d& input);
        void backprop(const Tensor4d& input);
        void backprop(const Tensor4d& input, Tensor4d& deda_lm1);
        void updateweights();

        /* setters */
        void set_a (const Tensor4d& data);

        /* getters */
        const Tensor4d& get_a () const { return a_; };
        UINT get_nneu () const { return a_.get_kernel(0, 0).width(); };
        UINT get_nneu_lm1 () const { return weights_.get_kernel(0, 0).height(); };
        typename layer::elayertype get_ltype() { return this->eFullLayer; };

    protected:
        void reset();
        void active_function_ (const Tensor4d& sum);
        void der_active_function_ ();
        float activate (float input) { return input; };
        float der_activate (float input) { return 1.0; };
        UINT num_neurons_;
        UINT num_samples_;
        float learning_rate_;
    };
}

#endif /* FLAYER_H_ */
