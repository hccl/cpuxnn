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

#include "matrix.h"

namespace xnn {

    class flayer {
    public:
    	flayer();
        flayer(const UINT nsamples, const UINT nneurons, bool is_trainable, const std::string name);
        flayer(const UINT nsamples, const UINT nneurons, UINT nneurons_lm1, bool is_trainable, const std::string name);
        flayer(const Matrix& weights, const Matrix& biases, bool is_trainable, const std::string name);
        virtual ~flayer();

        /* utilities */
        virtual void initial();
        virtual void propagate(const Matrix& input);
        virtual void backprop(const Matrix& input, Matrix& deda_prev);
        virtual void computegrad(const Matrix& input) {};
        virtual void updateweights();

        /* setters */
        void set_a (const Matrix& data);

        /* getters */
        const Matrix& get_a () const { return a_; };
        Matrix& get_deda () { return de_da_; };
        Matrix& get_dedw () { return de_dw_; };
        Matrix& get_dedb () { return de_db_; };
        bool get_flag () const { return is_trainable_; };
        UINT get_nneu () const { return a_.width(); };
        UINT get_nneu_lm1 () const { return weights_.height(); };
        std::string get_name () const { return name_; };
    protected:
        void reset();
        void active_function_ (const Matrix& sum);
        void der_active_function_ ();
        float activate (float input) { return input; };
        float der_activate (float input) { return 1.0; };
        Matrix a_;             /* neuron activations */
        Matrix weights_;       /* weight */
        Matrix biases_;        /* bias */
        Matrix de_dw_;         /* weight gradient, #input x #neurons
                                used as accumulator in parallel training */
        Matrix de_db_;         /* bias gradient, #neurons x 1
                                used as accumulator in parallel training */
        Matrix de_da_;         /* error of current layer */
        Matrix delta_;
        bool is_trainable_;
        UINT num_neurons_;
        UINT num_samples_;
        float learning_rate_;
        std::string name_;
    };
}

#endif /* FLAYER_H_ */
