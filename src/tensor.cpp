/*
 * Copyright 2014 by Institute of Acoustics, Chinese Academy of Sciences
 * http://www.ioa.ac.cn
 *
 * See the file COPYING for the licence associated with this software.
 *
 * Author(s):
 *   Xingyu Na, August 2014
 */

#include "tensor.h"

using namespace std;

namespace xnn {
    Tensor4d::Tensor4d() {
        dims_[0] = 0;
        dims_[1] = 0;
        dims_[2] = 0;
        dims_[3] = 0;
        data_ = NULL;
    }

    Tensor4d::Tensor4d(const Tensor4d & tnr) {
        dims_[0] = tnr.get_dim(0);
        dims_[1] = tnr.get_dim(1);
        dims_[2] = tnr.get_dim(2);
        dims_[3] = tnr.get_dim(3);
        if(data_ == NULL)
            data_ = new Matrix[dims_[0] * dims_[1]];
        this->copy(tnr);
    }

    Tensor4d::Tensor4d(UINT height, UINT width, UINT kernel_height, UINT kernel_width) {
        dims_[0] = height;
        dims_[1] = width;
        dims_[2] = kernel_height;
        dims_[3] = kernel_width;
        data_ = new Matrix[height * width];
        for(UINT i = 0; i < height * width; ++i) {
            data_[i] = Matrix(kernel_height, kernel_width);
            memset(data_[i].get_data(), 0, kernel_height * kernel_width * sizeof(float));
        }
    }

    Tensor4d::~Tensor4d() {
        delete[] data_;
        data_ = NULL;
    }

    void Tensor4d::tnrProductValid(const Tensor4d & lhs, const Tensor4d & rhs) {
        assert(lhs.get_dim(1) == rhs.get_dim(0));
        assert(dims_[0] == lhs.get_dim(0));
        assert(dims_[1] == rhs.get_dim(1));
        assert(dims_[2] == lhs.get_dim(2) - rhs.get_dim(2) + 1);
        assert(dims_[3] == lhs.get_dim(3) - rhs.get_dim(3) + 1);

        Matrix *sum = new Matrix(dims_[2], dims_[3]);
        Matrix *mul = new Matrix(dims_[2], dims_[3]);
        for (UINT i = 0; i < lhs.get_dim(0); ++i)
            for (UINT j = 0; j < rhs.get_dim(1); ++j) {
                sum->zeros_elt();
                for (UINT k = 0; k < lhs.get_dim(1); ++k) {
                    mul->matConvValid(lhs.get_kernel(i, k), rhs.get_kernel(k, j));
                    sum->matAdd(*mul);
                }
                this->get_kernel(i, j).copy(*sum);
            }
        delete mul; mul = NULL;
        delete sum; sum = NULL;
    }

    void Tensor4d::tnrProductFull(const Tensor4d & lhs, const Tensor4d & rhs) {
        assert(lhs.get_dim(1) == rhs.get_dim(0));
        assert(dims_[0] == lhs.get_dim(0));
        assert(dims_[1] == rhs.get_dim(1));
        assert(dims_[2] == lhs.get_dim(2) + rhs.get_dim(2) - 1);
        assert(dims_[3] == lhs.get_dim(3) + rhs.get_dim(3) - 1);

        Matrix *sum = new Matrix(dims_[2], dims_[3]);
        Matrix *mul = new Matrix(dims_[2], dims_[3]);
        for (UINT i = 0; i < lhs.get_dim(0); ++i)
            for (UINT j = 0; j < rhs.get_dim(1); ++j) {
                sum->zeros_elt();
                for (UINT k = 0; k < lhs.get_dim(1); ++k) {
                    mul->matConvFull(lhs.get_kernel(i, k), rhs.get_kernel(k, j));
                    sum->matAdd(*mul);
                }
                this->get_kernel(i, j).copy(*sum);
            }
        delete mul; mul = NULL;
        delete sum; sum = NULL;
    }

    void Tensor4d::tnrAdd(const Tensor4d & rhs) {
        assert(dims_[0] == rhs.dims_[0]);
        assert(dims_[1] == rhs.dims_[1]);
        assert(dims_[2] == rhs.dims_[2]);
        assert(dims_[3] == rhs.dims_[3]);

        Matrix *msum = new Matrix(dims_[2], dims_[3]);
        for (UINT i = 0; i < dims_[0]; ++i)
            for (UINT j = 0; j < dims_[1]; ++j) {
                msum->copy(this->get_kernel(i, j));
                msum->matAdd(rhs.get_kernel(i, j));
                this->get_kernel(i, j).copy(*msum);
            }
        delete msum; msum = NULL;
    }

    void Tensor4d::tnrSum(const Tensor4d & lhs, UINT idx) {
        float sum;
        if (idx == 0) {
            /* sum each row of matrices, along 2nd dim */
            assert(dims_[0] == lhs.get_dim(0));
            assert(dims_[1] == 1);
            assert(dims_[2] == 1);
            assert(dims_[3] == 1);
            Matrix *vsum = new Matrix(lhs.dims_[2], 1);
            for (UINT i = 0; i < dims_[0]; ++i) {
                sum = 0.0;
                for (UINT j = 0; j < lhs.get_dim(1); ++j) {
                    vsum->matSum(lhs.get_kernel(i, j), 0);
                    sum += vsum->vecSum();
                }
                this->set_elt(i, 0, 0, 0, sum);
            }
            delete vsum; vsum = NULL;
        } else if (idx == 1) {
            /* sum each column of matrices, along 1st dim */
            assert(dims_[0] == 1);
            assert(dims_[1] == lhs.get_dim(1));
            assert(dims_[2] == 1);
            assert(dims_[3] == 1);
            Matrix *vsum = new Matrix(lhs.dims_[2], 1);
            for (UINT i = 0; i < dims_[1]; ++i) {
                sum = 0.0;
                for (UINT j = 0; j < lhs.get_dim(0); ++j) {
                    vsum->matSum(lhs.get_kernel(j, i), 0);
                    sum += vsum->vecSum();
                }
                this->set_elt(0, i, 0, 0, sum);
            }
            delete vsum; vsum = NULL;
        }
    }

    void Tensor4d::eltProduct(const Tensor4d & rhs) {
        assert(dims_[0] == rhs.get_dim(0));
        assert(dims_[1] == rhs.get_dim(1));
        assert(dims_[2] == rhs.get_dim(2));
        assert(dims_[3] == rhs.get_dim(3));

        for (UINT i = 0; i < dims_[0]; ++i)
            for (UINT j = 0; j < dims_[1]; ++j)
                get_kernel(i, j).eltProduct(rhs.get_kernel(i, j));
    }        

    void Tensor4d::slrProduct(const float scalar) {
        /* simple implementation for test */
        for (UINT i = 0; i < dims_[0]; ++i)
            for (UINT j = 0; j < dims_[1]; ++j)
                get_kernel(i, j).slrProduct(scalar);
    }

    void Tensor4d::transpose(const Tensor4d & lhs) {
        assert(dims_[0] == lhs.get_dim(1));
        assert(dims_[1] == lhs.get_dim(0));
        assert(dims_[2] == lhs.get_dim(2));
        assert(dims_[3] == lhs.get_dim(3));
        for (UINT i = 0; i < dims_[0]; ++i)
            for (UINT j = 0; j < dims_[1]; ++j) {
                this->get_kernel(i, j).copy(lhs.get_kernel(j, i));
            }
    }

    void Tensor4d::flatten(Matrix & mat) const {
        assert(mat.height() == dims_[0]);
        assert(mat.width() == dims_[1] * dims_[2] * dims_[3]);
        for(UINT i = 0; i < dims_[0]; ++i)
            for(UINT j = 0; j < dims_[1]; ++j)
                for(UINT m = 0; m < dims_[2]; ++m)
                    for(UINT n = 0; n < dims_[3]; ++n)
                        mat.set_elt(i, j * dims_[2] * dims_[3] + m * dims_[3] + n, get_elt(i, j, m, n));
    }

    void Tensor4d::tensorize(const Matrix & mat) {
        assert( dims_[1] * dims_[2] * dims_[3] == mat.width());
        assert( dims_[0] == mat.height() );
        for(UINT i = 0; i < dims_[0]; ++i)
            for(UINT j = 0; j < dims_[1]; ++j)
                for(UINT m = 0; m < dims_[2]; ++m)
                    for(UINT n = 0; n < dims_[3]; ++n)
                        set_elt(i, j, m, n, mat.get_elt(i, j * dims_[2] * dims_[3] + m * dims_[3] + n));
    }

    void Tensor4d::copy(const Tensor4d & lhs) {
        for (UINT i = 0; i < dims_[0]; ++i)
            for (UINT j = 0; j < dims_[1]; ++j)
                get_kernel(i, j).copy(lhs.get_kernel(i, j));
    }

    void Tensor4d::zeros_elt() {
        for (UINT i = 0; i < dims_[0]; ++i)
            for (UINT j = 0; j < dims_[1]; ++j)
                get_kernel(i, j).zeros_elt();
    }

    void Tensor4d::ones_elt() {
        for (UINT i = 0; i < dims_[0]; ++i)
            for (UINT j = 0; j < dims_[1]; ++j)
                get_kernel(i, j).ones_elt();
    }

    void Tensor4d::rand_elt() {
        for (UINT i = 0; i < dims_[0]; ++i)
            for (UINT j = 0; j < dims_[1]; ++j)
                get_kernel(i, j).rand_elt();
    }

    Matrix & Tensor4d::get_kernel(UINT i, UINT j) const {
        return this->data_[i * this->dims_[1] + j];
    }

    float Tensor4d::get_elt(UINT i, UINT j, UINT m, UINT n) const {
        return this->get_kernel(i, j).get_elt(m, n);
    }

    void Tensor4d::set_elt(UINT i, UINT j, UINT m, UINT n, float f) {
        this->get_kernel(i, j).set_elt(m, n, f);
    }

    void Tensor4d::set_kernel(UINT i, UINT j, Matrix & mat) {
        this->data_[i * this->dims_[1] +j] = mat;
    }

    /* operators */
    Tensor4d & Tensor4d::operator = (const Tensor4d & rhs) {
        if (this == &rhs)
            return *this;
        this->dims_[0] = rhs.dims_[0];
        this->dims_[1] = rhs.dims_[1];
        this->dims_[2] = rhs.dims_[2];
        this->dims_[3] = rhs.dims_[3];
        delete[] data_;
        data_ = new Matrix[dims_[0] * dims_[1]];
        for(UINT i = 0; i < dims_[0] * dims_[1]; ++i) {
            data_[i] = Matrix(dims_[2], dims_[3]);
            memcpy(this->data_[i].get_data(), rhs.data_[i].get_data(), rhs.dims_[2] * rhs.dims_[3] * sizeof(float));
        }
        return *this;
    }

    void Tensor4d::display() const {
        UINT height = dims_[0];
        UINT width = dims_[1];
        UINT kernel_height = dims_[2];
        UINT kernel_width = dims_[3];
        cout.precision(4);
        for(UINT i = 0; i < height * kernel_height; ++i) {
            if(i%kernel_height == 0)
                cout << endl;
            for(UINT j = 0; j < width * kernel_width; ++j) {
                if (j%kernel_width == 0)
                    cout << " ";
                cout << " " << this->get_elt(i/kernel_height, j/kernel_width, i%kernel_height, j%kernel_width);
            }
            cout << endl;
        }
    }

}
