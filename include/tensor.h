/*
 * Copyright 2014 by Institute of Acoustics, Chinese Academy of Sciences
 * http://www.ioa.ac.cn
 *
 * See the file COPYING for the licence associated with this software.
 *
 * Author(s):
 *   Xingyu Na, June 2014
 */

#ifndef _TENSOR_H
#define _TENSOR_H

#include "common.h"
#include "matrix.h"

using namespace std;

namespace xnn {

    /* implement 4-d tensor as matrix of matrices */
    class Tensor4d {
    public:
        Tensor4d();
        Tensor4d(const Tensor4d & tnr);
        Tensor4d(UINT height, UINT width, UINT kernel_height, UINT kernel_width);
        ~Tensor4d();
        void tnrProductValid(const Tensor4d & lhs, const Tensor4d & rhs);
        void tnrProductFull(const Tensor4d & lhs, const Tensor4d & rhs);
        void tnrAdd(const Tensor4d & rhs);
        void tnrSum(const Tensor4d & lhs, UINT idx);
        void eltProduct(const Tensor4d & rhs);
        void slrProduct(const float scalar);
        void transpose(const Tensor4d & lhs);
        void tensorize(const Matrix & mat);
        void flatten(Matrix & mat) const;
        void copy(const Tensor4d & lhs);
        void zeros_elt();
        void ones_elt();
        void rand_elt();
        bool is_flatten() { return false; };
        void display();

        /* operators */
        Tensor4d & operator = (const Tensor4d & rhs);
        void set_kernel(UINT i, UINT j, Matrix & mat);
        Matrix & get_kernel(UINT i, UINT j) const;
        void set_elt(UINT i, UINT j, UINT m, UINT n, float f);
        float get_elt(UINT i, UINT j, UINT m, UINT n) const;
        UINT get_dim(UINT d) const { return dims_[d]; };
    protected:
        UINT dims_[4];
        Matrix *data_;
    };
}

#endif                    /* _TENSOR_H */
