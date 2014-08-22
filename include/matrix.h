/*
 * Copyright 2014 by Institute of Acoustics, Chinese Academy of Sciences
 * http://www.ioa.ac.cn
 *
 * See the file COPYING for the licence associated with this software.
 *
 * Author(s):
 *   Xingyu Na, July 2014
 */

#ifndef _MATRIX_H_
#define _MATRIX_H_

#include "common.h"

namespace xnn {

    class Matrix {
    public:
        Matrix();
        Matrix(const Matrix & mat);
        Matrix(UINT height, UINT width);
        ~Matrix();
        void matProduct(const Matrix & lhs, const Matrix & rhs);
        void matConvValid(const Matrix & lhs, const Matrix & rhs);
        void matConvFull(const Matrix & lhs, const Matrix & rhs);
        void matAdd(const Matrix & rhs);
        void matMinus(const Matrix & rhs);
        void eltProduct(const Matrix & rhs);
        void slrProduct(const float & scalar);
        void slrAdd(const float & scalar);
        void matSum(const Matrix & lhs, const UINT idx);
        float vecSum();
        void transpose(const Matrix & lhs);
        void samplemax(const Matrix & lhs, UINT sampler_height, UINT sampler_width);
        void sampleavg(const Matrix & lhs, UINT sampler_height, UINT sampler_width);
        void unsamplemax(const Matrix & lhs, const Matrix & input, const Matrix & output, UINT sampler_height, UINT sampler_width);
        void unsampleavg(const Matrix & lhs, UINT sampler_height, UINT sampler_width);
        void copy(const Matrix & lhs);
        void zeros_elt();
        void ones_elt();
        void rand_elt();
        UINT width() const { return this->width_; };
        UINT height() const { return this->height_; };
        bool writeRow(UINT i, UINT len, float *ptr);
        void display() const;
        void print_size() const;
        bool is_flatten() { return true; };

        /* operators */
        Matrix & operator = (const Matrix & rhs);
        void set_elt(UINT i, UINT j, float f);
        float get_elt(UINT i, UINT j) const;
        float * get_data() const { return data_; };
    private:
        float *data_;
        UINT width_;
        UINT height_;
    };
}

#endif				/* _MATRIX_H_ */
