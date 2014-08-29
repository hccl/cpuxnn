/*
 * Copyright 2014 by Institute of Acoustics, Chinese Academy of Sciences
 * http://www.ioa.ac.cn
 *
 * See the file COPYING for the licence associated with this software.
 *
 * Author(s):
 *   Zhichao Wang, August 2014
 */

#include "neuron.h"

Neuron::Neuron()
{
}

Neuron::~Neuron()
{
}

void Neuron::init(enumFuncType funcType) {
    funcType_ = funcType;
};

float Neuron::act_Sigmoid(float z) {
    return 1/(1+expf(-z));
}

float Neuron::der_Sigmoid(float a) {
    return a*(1-a);
}

float Neuron::act_Tanh(float z) {
    return (expf(z)-expf(-z))/(expf(z)+exp(-z));
}

float Neuron::der_Tanh(float a) {
    return a*(1-a);
}

float Neuron::act_Linear(float z) {
    return z;
}

float Neuron::der_Linear(float a) {
    return 1.0f;
}

float Neuron::act_Relu(float z) {
    return z > 0.0f ? z : 0.0f;
}

float Neuron::der_Relu(float a) {
    return a > 0.0f ? 1 : 0.0f;
}

float Neuron::activate(float z) {
    switch (funcType_){
    case SIGMOID:
        return act_Sigmoid(z);
        break;
    case TANH:
        return act_Tanh(z);
        break;
    case LINEAR:
        return act_Linear(z);
        break;
    case RELU:
        return act_Relu(z);
        break;
    default:
        throw runtime_error("Unknown function type");
    }
}

float Neuron::der_activate(float a) {
    switch (funcType_){
    case SIGMOID:
        return der_Sigmoid(a);
        break;
    case TANH:
        return der_Tanh(a);
        break;
    case LINEAR:
        return der_Linear(a);
        break;
    case RELU:
        return der_Relu(a);
        break;
    default:
        throw runtime_error("Unknown function type");
    }
    
}
