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

    flayer::flayer(const UINT nsamples, const UINT nneurons, bool is_trainable, const std::string name, const enumFuncType actType) {
        num_samples_ = nsamples;
        num_neurons_ = nneurons;
        a_ = Tensor4d(1, 1, nsamples, nneurons);
        is_trainable_ = is_trainable;
        name_ = name;
        funcType_ = actType;
        neurons_.init(funcType_);
    }

    flayer::flayer(const UINT nsamples, const UINT nneurons, const UINT nneurons_lm1, bool is_trainable, const std::string name, const enumFuncType actType) {
        num_samples_ = nsamples;
        num_neurons_ = nneurons;
        a_ = Tensor4d(1, 1, nsamples, nneurons);
        de_da_ = Tensor4d(1, 1, nsamples, nneurons);
        delta_ = Tensor4d(1, 1, nsamples, nneurons);
        weights_ = Tensor4d(1, 1, nneurons_lm1, nneurons);
        de_dw_ = Tensor4d(1, 1, nneurons_lm1, nneurons);
        biases_ = Tensor4d(1, 1, 1, nneurons);
        de_db_ = Tensor4d(1, 1, 1, nneurons);
        is_trainable_ = is_trainable;
        name_ = name;
        funcType_ = actType;
        neurons_.init(funcType_);
    }

    flayer::flayer(const Tensor4d& weights, const Tensor4d& biases, bool is_trainable, const std::string name, const enumFuncType actType) {
        num_neurons_ = weights.get_kernel(0, 0).width();
        num_samples_ = weights.get_kernel(0, 0).height();
        /* this is not right, fix it. */
        a_ = Tensor4d(1, 1, num_samples_, num_neurons_);
        weights_ = weights;
        biases_ = biases;
        is_trainable_ = is_trainable;
        name_ = name;
        funcType_ = actType;
        neurons_.init(funcType_);
    }

    void flayer::initial() {
        weights_.rand_elt();
        biases_.rand_elt();
        learning_rate_ = 0.08;
    }

    void flayer::propagate(const Tensor4d& input) {
        Tensor4d *z = new Tensor4d (1, 1, a_.get_kernel(0, 0).height(), a_.get_kernel(0, 0).width());
        Matrix *eye = new Matrix (input.get_kernel(0, 0).height(), 1);
        Matrix *b = new Matrix (a_.get_kernel(0, 0).height(), a_.get_kernel(0, 0).width());
        assert(input.get_kernel(0, 0).width() == this->weights_.get_kernel(0, 0).height());

        z->get_kernel(0, 0).matProduct(input.get_kernel(0, 0), weights_.get_kernel(0, 0));
        eye->ones_elt();
        b->matProduct(*eye, biases_.get_kernel(0, 0));
        z->get_kernel(0, 0).matAdd(*b);
        if (name_ == "output")
        {
            softmax_function_(*z);
        }else
        {
            active_function_(*z);
        }

        delete b; b = NULL;
        delete eye; eye = NULL;
        delete z; z = NULL;
    }

    void flayer::backprop(const Tensor4d& input) {
        Matrix *transp = new Matrix(input.get_kernel(0, 0).width(), input.get_kernel(0, 0).height());
        if (name_ == "output")
        {
            der_softmax_function_();
        }else
        {
            der_active_function_();
        } 
        delta_.eltProduct(this->de_da_);
        transp->transpose(input.get_kernel(0, 0));
        de_dw_.get_kernel(0, 0).matProduct(*transp, delta_.get_kernel(0, 0));
        de_db_.get_kernel(0, 0).matSum(delta_.get_kernel(0, 0), 1);
        delete transp; transp = NULL;
    }

    void flayer::backprop(const Tensor4d& input, Tensor4d& deda_lm1) {
        Matrix *transp = new Matrix(input.get_kernel(0, 0).width(), input.get_kernel(0, 0).height());
        Matrix *wtransp = new Matrix(weights_.get_kernel(0, 0).width(), weights_.get_kernel(0, 0).height());
        if (name_ == "output")
        {
            der_softmax_function_();
        }else
        {
            der_active_function_();
        } 
        delta_.eltProduct(this->de_da_);
        transp->transpose(input.get_kernel(0, 0));
        de_dw_.get_kernel(0, 0).matProduct(*transp, delta_.get_kernel(0, 0));
        de_db_.get_kernel(0, 0).matSum(delta_.get_kernel(0, 0), 1);
        wtransp->transpose(weights_.get_kernel(0, 0));
        deda_lm1.get_kernel(0, 0).matProduct(delta_.get_kernel(0, 0), *wtransp);
        delete transp; transp = NULL;
        delete wtransp; wtransp = NULL;
    }

    void flayer::updateweights() {
        de_dw_.get_kernel(0, 0).slrProduct(learning_rate_);
        de_db_.get_kernel(0, 0).slrProduct(learning_rate_);
        weights_.get_kernel(0, 0).matAdd(de_dw_.get_kernel(0, 0));
        biases_.get_kernel(0, 0).matAdd(de_db_.get_kernel(0, 0));
    }

    void flayer::set_a (const Tensor4d& data) {
        a_.copy(data);
    }

    void flayer::active_function_ ( const Tensor4d& sum ) {
        assert(this->a_.get_kernel(0, 0).width() == sum.get_kernel(0, 0).width());
        assert(this->a_.get_kernel(0, 0).height() == sum.get_kernel(0, 0).height());
        for(UINT i = 0; i < sum.get_kernel(0, 0).height(); ++i) {
            for(UINT j = 0; j < sum.get_kernel(0, 0).width(); ++j) {
                a_.set_elt(0, 0, i, j, neurons_.activate(sum.get_elt(0, 0, i, j)));
            }
        }
    }

    void flayer::softmax_function_ (const Tensor4d& sum){
        assert(this->a_.get_kernel(0, 0).width() == sum.get_kernel(0, 0).width());
        assert(this->a_.get_kernel(0, 0).height() == sum.get_kernel(0, 0).height());    
        float sumRow = 0.0f;
        for(UINT i = 0; i < sum.get_kernel(0, 0).height(); ++i) {
            for(UINT j = 0; j < sum.get_kernel(0, 0).width(); ++j) {
                a_.set_elt(0, 0, i, j, expf(sum.get_elt(0, 0, i, j)));
                sumRow += a_.get_elt(0, 0, i, j);
            }
            for(UINT j = 0; j < sum.get_kernel(0, 0).width(); j++) {
                a_.set_elt(0, 0, i, j, a_.get_elt(0, 0, i, j)/sumRow);
            }
            sumRow = 0.0f;
        }
    }

    void flayer::der_active_function_ () {
        for(UINT i = 0; i < a_.get_kernel(0, 0).height(); ++i)
            for(UINT j = 0; j < a_.get_kernel(0, 0).width(); ++j)
                delta_.set_elt(0, 0, i, j, neurons_.der_activate(a_.get_elt(0, 0, i, j)));
    }

    void flayer::der_softmax_function_ () {
        for(UINT i = 0; i < a_.get_kernel(0, 0).height(); ++i) {
            for(UINT j = 0; j < a_.get_kernel(0, 0).width(); ++j) {
                delta_.set_elt(0, 0, i, j, der_softmax(a_.get_elt(0, 0, i, j)));
            }
        }    
    }

}
