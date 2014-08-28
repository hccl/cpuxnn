/*
 * Copyright 2014 by Institute of Acoustics, Chinese Academy of Sciences
 * http://www.ioa.ac.cn
 *
 * See the file COPYING for the licence associated with this software.
 *
 * Author(s):
 *   Xingyu Na, August 2014
 */

#include "common.h"
#include "matrix.h"

namespace xnn {
    Matrix::Matrix() {
        width_ = 0;
        height_ = 0;
        data_ = NULL;
    }

    Matrix::Matrix(const Matrix & mat) {
        height_ = mat.height();
        width_ = mat.width();
        if(data_ == NULL)
            data_ = new float[height_ * width_];
        this->copy(mat);
    }

    Matrix::Matrix(UINT height, UINT width) {
        width_ = width;
        height_ = height;
        data_ = new float[width * height];
        memset(data_, 0, width * height * sizeof(float));
    }

    Matrix::~Matrix() {
        delete [] data_;
        data_ = NULL;
    }

    void Matrix::matProduct(const Matrix & lhs, const Matrix & rhs) {
        assert(lhs.width() == rhs.height());
        assert(height_ == lhs.height());
        assert(width_ == rhs.width());
        float sum;
	/* simple implementation for test */
        for (UINT i = 0; i < height_; ++i)
            for (UINT j = 0; j < width_; ++j) {
                sum = 0.0;
                for (UINT k = 0; k < lhs.width(); ++k)
                    sum += lhs.get_elt(i, k) * rhs.get_elt(k, j);
                this->set_elt(i, j, sum);
            }
    }

    void Matrix::matConvValid(const Matrix & lhs, const Matrix & rhs) {
        assert(lhs.width() >= rhs.width());
        assert(lhs.height() >= rhs.height());
        UINT h_shift = lhs.height() - rhs.height() + 1;
        UINT w_shift = lhs.width() - rhs.width() + 1;
        assert(height_ == h_shift);
        assert(width_ == w_shift);
        float sum;
        for(UINT i = 0; i < h_shift; ++i)
            for(UINT j = 0; j < w_shift; ++j) {
                sum = 0.0;
                for(UINT m = 0; m < rhs.height(); ++m)
                    for(UINT n = 0; n < rhs.width(); ++n)
                        sum += lhs.get_elt(i + rhs.height() - 1 - m, j + rhs.width() - 1 - n)
                            * rhs.get_elt(m, n);
                this->set_elt(i, j, sum);
            }
    }

    void Matrix::matConvFull(const Matrix & lhs, const Matrix & rhs) {
        UINT h_shift = lhs.height() + rhs.height() - 1;
        UINT w_shift = lhs.width() + rhs.width() - 1;
        assert(height_ == h_shift);
        assert(width_ == w_shift);
        float sum;
        for(UINT i = 0; i < h_shift; ++i)
            for(UINT j = 0; j < w_shift; ++j) {
                sum = 0.0;
                for(UINT m = 0; m < rhs.height(); ++m)
                    for(UINT n = 0; n < rhs.width(); ++n)
                        if (i - m >= 0 && i - m < height_ && j - n >= 0 && j - n < width_)
                            sum += lhs.get_elt(i - m, j - n) * rhs.get_elt(m, n);
                this->set_elt(i, j, sum);
            }
    }

    void Matrix::matAdd(const Matrix & rhs) {
        assert(width_ == rhs.width());
        assert(height_ == rhs.height());
        /* simple implementation for test */
        for (UINT i = 0; i < height_; ++i)
            for (UINT j = 0; j < width_; ++j)
                this->set_elt(i, j, this->get_elt(i, j) + rhs.get_elt(i, j));
    }

    void Matrix::matMinus(const Matrix & rhs) {
        assert(width_ == rhs.width());
        assert(height_ == rhs.height());
        /* simple implementation for test */
        for (UINT i = 0; i < height_; ++i)
            for (UINT j = 0; j < width_; ++j)
                this->set_elt(i, j, this->get_elt(i, j) - rhs.get_elt(i, j));
    }

    void Matrix::eltProduct(const Matrix & rhs) {
        assert(width_ == rhs.width());
        assert(height_ == rhs.height());
        /* simple implementation for test */
        for (UINT i = 0; i < height_; ++i)
            for (UINT j = 0; j < width_; ++j)
                this->set_elt(i, j, this->get_elt(i, j) * rhs.get_elt(i, j));
    }

    void Matrix::slrProduct(const float & scalar) {
        UINT i, j;
        /* simple implementation for test */
        for (i = 0; i < height_; ++i)
            for (j = 0; j < width_; ++j)
                this->set_elt(i, j, this->get_elt(i, j) * scalar);
    }

    void Matrix::slrAdd(const float & scalar) {
        UINT i, j;
        /* simple implementation for test */
        for (i = 0; i < height_; ++i)
            for (j = 0; j < width_; ++j)
                this->set_elt(i, j, this->get_elt(i, j) + scalar);
    }

    void Matrix::matSum(const Matrix & lhs, const UINT idx) {
        float sum;
        if (idx == 0) {
            /* sum each row, along width */
            assert(height_ == lhs.height());
            assert(width_ == 1);
            for (UINT i = 0; i < height_; ++i) {
                sum = 0.0;
                for (UINT j = 0; j < lhs.width(); ++j)
                    sum += lhs.get_elt(i, j);
                this->set_elt(i, 0, sum);
            }
        } else if (idx == 1) {
            /* sum each column, along height */
            assert(width_ == lhs.width());
            assert(height_ == 1);
            for (UINT i = 0; i < width_; ++i) {
                sum = 0.0;
                for (UINT j = 0; j < height_; ++j)
                    sum += lhs.get_elt(j, i);
                this->set_elt(0, i, sum);
            }
        }
    }

    float Matrix::vecSum() {
        float sum = 0.0;
        if(width_ == 1)
            for(UINT i = 0; i < height_; ++i)
                sum += get_elt(i, 0);
        else if (height_ == 1)
            for(UINT i = 0; i < width_; ++i)
                sum += get_elt(0, i);
        return sum;
    }

    void Matrix::transpose(const Matrix & lhs) {
        assert(height_ == lhs.width());
        assert(width_ == lhs.height());
        for (UINT i = 0; i < height_; ++i)
            for (UINT j = 0; j < width_; ++j)
                this->set_elt(i, j, lhs.get_elt(j, i));
    }

    void Matrix::samplemax(const Matrix & lhs, UINT sampler_height, UINT sampler_width) {
        assert(lhs.height() % sampler_height == 0);
        assert(lhs.width() % sampler_width == 0);
        UINT h = lhs.height() / sampler_height, w = lhs.width() / sampler_width;
        assert(height_ == h);
        assert(width_ == w);
        float fmax, ftemp;
        for( UINT i = 0; i < h; ++i )
            for ( UINT j = 0; j < w; ++j ) {
                fmax = FMIN_VALUE;
                for (UINT m = 0; m < sampler_height; ++m)
                    for (UINT n = 0; n < sampler_width; ++n) {
                        ftemp = lhs.get_elt(i * sampler_height + m, j * sampler_width + n);
                        if (ftemp > fmax)
                            fmax = ftemp;
                    }
                this->set_elt(i, j, fmax);
            }
    }

    void Matrix::sampleavg(const Matrix & lhs, UINT sampler_height, UINT sampler_width ) {
        assert(lhs.height() % sampler_height == 0);
        assert(lhs.width() % sampler_width == 0);
        UINT h = lhs.height() / sampler_height, w = lhs.width() / sampler_width;
        assert(height_ == h);
        assert(width_ == w);
        float fsum;
        for( UINT i = 0; i < h; ++i )
            for ( UINT j = 0; j < w; ++j ) {
                fsum = 0.0;
                for (UINT m = 0; m < sampler_height; ++m)
                    for (UINT n = 0; n < sampler_width; ++n)
                        fsum += lhs.get_elt(i * sampler_height + m, j * sampler_width + n);
                this->set_elt(i, j, fsum / (sampler_height * sampler_width));
            }
    }

    void Matrix::unsamplemax(const Matrix & lhs, const Matrix & input, const Matrix & output,
                             UINT sampler_height, UINT sampler_width ) {
        UINT h = lhs.height() * sampler_height, w = lhs.width() * sampler_width;
        assert(height_ == h);
        assert(width_ == w);
        assert(input.height() == h);
        assert(input.width() == w);
        float fsample, finput, fout;
        for( UINT i = 0; i < lhs.height(); ++i )
            for ( UINT j = 0; j < lhs.width(); ++j ) {
                fsample = lhs.get_elt(i, j);
                fout = output.get_elt(i, j);
                for (UINT m = 0; m < sampler_height; ++m)
                    for (UINT n = 0; n < sampler_width; ++n) {
                        finput = input.get_elt(i * sampler_height + m, j * sampler_width + n);
                        if (finput == fout)
                            this->set_elt(i * sampler_height + m, j * sampler_width + n, fsample);
                        else
                            this->set_elt(i * sampler_height + m, j * sampler_width + n, 0.0);
                    }
            }
    }

    void Matrix::unsampleavg(const Matrix & lhs, UINT sampler_height, UINT sampler_width ) {
        UINT h = lhs.height() * sampler_height, w = lhs.width() * sampler_width;
        assert(height_ == h);
        assert(width_ == w);
        float fsample;
        for( UINT i = 0; i < lhs.height(); ++i )
            for ( UINT j = 0; j < lhs.width(); ++j ) {
                fsample = lhs.get_elt(i, j) / (sampler_height * sampler_width);
                for (UINT m = 0; m < sampler_height; ++m)
                    for (UINT n = 0; n < sampler_width; ++n)
                        this->set_elt(i * sampler_height + m, j * sampler_width + n, fsample);
            }
    }

    void Matrix::copy(const Matrix & lhs) {
        assert(height_ == lhs.height());
        assert(width_ == lhs.width());
        for (UINT i = 0; i < height_; ++i)
            for (UINT j = 0; j < width_; ++j)
                this->set_elt(i, j, lhs.get_elt(i, j));
    }

    void Matrix::zeros_elt() {
        for (UINT i = 0; i < height_; ++i)
            for (UINT j = 0; j < width_; ++j)
                this->set_elt(i, j, 0.0);
    }

    void Matrix::ones_elt() {
        for (UINT i = 0; i < height_; ++i)
            for (UINT j = 0; j < width_; ++j)
                this->set_elt(i, j, 1.0);
    }

    void Matrix::rand_elt() {
        for (UINT i = 0; i < height_; ++i)
            for (UINT j = 0; j < width_; ++j)
                this->set_elt(i, j, (float (rand()) / RAND_MAX));
    }

    bool Matrix::writeRow(UINT i, UINT len, float *ptr) {
        assert(len == width_);
        for (UINT j = 0; j < width_; ++j)
            this->set_elt(i, j, *(ptr++));
        return true;
    }

    void Matrix::set_elt(UINT i, UINT j, float f) {
        this->data_[i * this->width() + j] = f;
    }

    float Matrix::get_elt(UINT i, UINT j) const {
        return this->data_[i * this->width() + j];
    }

    void Matrix::display() const {
        for (UINT i = 0; i < height_; ++i) {
            for (UINT j = 0; j < width_; ++j)
                std::cout << get_elt(i, j) << " ";
            std::cout << std::endl;
        }
    }

    void Matrix::print_size() const {
        std::cout << height_ << " x " << width_;
    }

	float* Matrix::getRowData(int row) const{
		return &data_[row*width_];
	}

    /* operators */
    Matrix & Matrix::operator = (const Matrix & rhs) {
        if (this == &rhs)
            return *this;
        this->width_ = rhs.width();
        this->height_ = rhs.height();
        delete [] data_;
        data_ = new float[width_ * height_];
        memcpy(this->data_, rhs.get_data(), width_ * height_ * sizeof(float));
        return *this;
    }
}
