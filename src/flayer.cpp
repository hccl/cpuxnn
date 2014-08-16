/*
 * Copyright 2014 by Institute of Acoustics, Chinese Academy of Sciences
 * http://www.ioa.ac.cn
 *
 * See the file COPYING for the licence associated with this software.
 *
 * Author(s):
 *   Xingyu Na, June 2014
 */

#include "flayer.h"

namespace xnn {
    flayer::flayer() {
        reset();
    }

    void flayer::reset() {
        num_neurons_ = 0;
        num_samples_ = 0;
        is_trainable_ = false;
    }

    flayer::~flayer() {
        reset();
    }

    flayer::flayer(const UINT nsamples, const UINT nneurons, bool is_trainable, const std::string name) {
        num_samples_ = nsamples;
        num_neurons_ = nneurons;
        a_ = Matrix(nsamples, nneurons);
        is_trainable_ = is_trainable;
        name_ = name;
    }

    flayer::flayer(const UINT nsamples, const UINT nneurons, const UINT nneurons_lm1, bool is_trainable, const std::string name) {
        num_samples_ = nsamples;
        num_neurons_ = nneurons;
        a_ = Matrix(nsamples, nneurons);
        de_da_ = Matrix(nsamples, nneurons);
        delta_ = Matrix(nsamples, nneurons);
        weights_ = Matrix(nneurons_lm1, nneurons);
        de_dw_ = Matrix(nneurons_lm1, nneurons);
        biases_ = Matrix(1, nneurons);
        de_db_ = Matrix(1, nneurons);
        is_trainable_ = is_trainable;
        name_ = name;
    }

    flayer::flayer(const Matrix& weights, const Matrix& biases, bool is_trainable, const std::string name) {
        num_neurons_ = weights.width();
        num_samples_ = weights.height();
        /* this is not right, fix it. */
        a_ = Matrix(num_samples_, num_neurons_);
        weights_ = weights;
        biases_ = biases;
        is_trainable_ = is_trainable;
        name_ = name;
    }

    void flayer::initial() {
        UINT i, j;
        for(i = 0; i < weights_.height(); ++i)
            for(j = 0; j < weights_.width(); ++j)
                weights_.set_elt(i, j, float(-0.5) + (float(rand()) / RAND_MAX));
        for(i = 0; i < biases_.height(); ++i)
            biases_.set_elt(i, 0, float(-0.5) + (float(rand()) / RAND_MAX));
        learning_rate_ = 0.001;
    }

    void flayer::propagate(const Matrix& input) {
        Matrix *z = new Matrix (a_.height(), a_.width());
        Matrix *eye = new Matrix (input.height(), 1);
        Matrix *b = new Matrix (a_.height(), a_.width());
        assert(input.width() == this->weights_.height());

        z->matProduct(input, weights_);
        eye->ones_elt();
        b->matProduct(*eye, biases_);
        z->matAdd(*b);
        active_function_(*z);

        delete b; b = NULL;
        delete eye; eye = NULL;
        delete z; z = NULL;
    }

    void flayer::backprop(const Matrix& input, Matrix& deda_lm1) {
        Matrix *transp = new Matrix(input.width(), input.height());
        Matrix *wtransp = new Matrix(weights_.width(), weights_.height());
        der_active_function_();
        delta_.eltProduct(this->de_da_);
        transp->transpose(input);
        de_dw_.matProduct(*transp, delta_);
        de_db_.matSum(delta_, 1);
        wtransp->transpose(weights_);
        deda_lm1.matProduct(delta_, *wtransp);
    }

    void flayer::updateweights() {
        de_dw_.slrProduct(learning_rate_);
        de_db_.slrProduct(learning_rate_);
        weights_.matAdd(de_dw_);
        biases_.matAdd(de_db_);
    }

    void flayer::set_a (const Matrix& data) {
        a_.copy(data);
    }

    void flayer::active_function_ ( const Matrix& sum ) {
        assert(this->a_.width() == sum.width());
        assert(this->a_.height() == sum.height());
        for(UINT i = 0; i < sum.height(); ++i) {
            for(UINT j = 0; j < sum.width(); ++j) {
                a_.set_elt(i, j, activate(sum.get_elt(i, j)));
            }
        }
    }

    void flayer::der_active_function_ () {
        for(UINT i = 0; i < a_.height(); ++i)
            for(UINT j = 0; j < a_.width(); ++j)
                delta_.set_elt(i, j, der_activate(a_.get_elt(i, j)));
    }

}
