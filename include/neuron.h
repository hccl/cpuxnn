/*
 * Copyright 2014 by Institute of Acoustics, Chinese Academy of Sciences
 * http://www.ioa.ac.cn
 *
 * See the file COPYING for the licence associated with this software.
 *
 * Author(s):
 *   Zhichao Wang, August 2014
 */

#ifndef _NEURON_H
#define _NEURON_H

#include "common.h"

class Neuron
{
public:
    Neuron();
    ~Neuron();
    void init(enumFuncType funcType);
    float activate(float z);
    float der_activate(float a);


private:
    enumFuncType funcType_;

    float act_Sigmoid(float z);
    float act_Tanh(float z);
    float act_Linear(float z);
    float act_Relu(float z);
    float der_Sigmoid(float a);
    float der_Tanh(float a);
    float der_Linear(float a);
    float der_Relu(float a);
};

#endif