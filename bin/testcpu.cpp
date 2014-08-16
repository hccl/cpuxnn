/*
 Read Me:
 This is a sanity test code for the dummy implementation of neural
 network training library ``cpuxnn.a'', supporting fully-connected
 layer, convolutional layer and pooling layer. The training method is
 gradient descent(GD). Batch GD and Stochastic GD (online and
 minibatch) can be implemented by calling the back propagation routine
 and weight updating routine in different ways. Second-order training
 techniques are in the Todo list.

 The codes are tested using GNU g++4.6

 Author(s): Xingyu Na, August 2014
 */


#include "clayer.h"
#include "flayer.h"
#include "player.h"

using namespace std;
using namespace xnn;

int main(void) {
    Tensor4d *data;
    Matrix *label;
    clayer* c;
    clayer* input;
    player* p;
    flayer* output;

    UINT nsamples = 5;
    UINT feature_height = 3;
    UINT feature_width = 5;
    UINT nfilter = 2;
    UINT filter_height = 3;
    UINT filter_width = 2;
    UINT sampler_height = 1;
    UINT sampler_width = 2;
    UINT nclass = 2;
    UINT epochs = 50;

    /* load data and label */
    data = new Tensor4d(nsamples, 1, feature_height, feature_width);
    float sample1[] = {2.1, 2.2, 0.2, 0.1, 0.2}; /* sample #1 */
    float sample2[] = {2.4, 2.8, 0.7, 0.3, 0.4};
    float sample3[] = {0.7, 0.3, 0.9, 0.2, 0.3};
    data->get_kernel(0, 0).writeRow(0, 5, sample1);
    data->get_kernel(0, 0).writeRow(1, 5, sample2);
    data->get_kernel(0, 0).writeRow(2, 5, sample3);

    float sample4[] = {0.2, 0.1, 0.4, 0.1, 0.2}; /* sample #2 */
    float sample5[] = {0.7, 0.3, 0.5, 2.4, 2.8};
    float sample6[] = {0.7, 0.3, 0.4, 2.9, 2.2};
    data->get_kernel(1, 0).writeRow(0, 5, sample4);
    data->get_kernel(1, 0).writeRow(1, 5, sample5);
    data->get_kernel(1, 0).writeRow(2, 5, sample6);

    float sample7[] = {0.2, 0.1, 0.3, 0.7, 0.3}; /* sample #3 */
    float sample8[] = {0.7, 0.3, 0.2, 2.2, 2.2};
    float sample9[] = {0.1, 0.2, 0.7, 2.9, 2.2};
    data->get_kernel(2, 0).writeRow(0, 5, sample7);
    data->get_kernel(2, 0).writeRow(1, 5, sample8);
    data->get_kernel(2, 0).writeRow(2, 5, sample9);

    float sample10[] = {2.2, 2.1, 0.2, 0.1, 0.2}; /* sample #4 */
    float sample11[] = {2.7, 2.3, 0.3, 0.4, 0.8};
    float sample12[] = {0.7, 0.3, 0.4, 0.9, 0.2};
    data->get_kernel(3, 0).writeRow(0, 5, sample10);
    data->get_kernel(3, 0).writeRow(1, 5, sample11);
    data->get_kernel(3, 0).writeRow(2, 5, sample12);

    float sample13[] = {2.5, 2.6, 0.3, 0.1, 0.2}; /* sample #5 */
    float sample14[] = {2.7, 2.2, 0.2, 0.4, 0.4};
    float sample15[] = {0.7, 0.3, 0.1, 0.9, 0.2};
    data->get_kernel(4, 0).writeRow(0, 5, sample13);
    data->get_kernel(4, 0).writeRow(1, 5, sample14);
    data->get_kernel(4, 0).writeRow(2, 5, sample15);

    label = new Matrix(nsamples, nclass);
    float label1[] = {1.0, 0.0};
    float label2[] = {0.0, 1.0};
    float label3[] = {0.0, 1.0};
    float label4[] = {1.0, 0.0};
    float label5[] = {1.0, 0.0};
    label->writeRow(0, 2, label1);
    label->writeRow(1, 2, label2);
    label->writeRow(2, 2, label3);
    label->writeRow(3, 2, label4);
    label->writeRow(4, 2, label5);

    /* network structure */
    assert((feature_height - filter_height + 1) % sampler_height == 0);
    assert((feature_width - filter_width + 1) % sampler_width == 0);

    input = new clayer(nsamples, 1, 1, 1, 1, feature_height, feature_width, "data");
    input->set_a(*data);

    c = new clayer(nsamples, nfilter, filter_height, filter_width, 1, feature_height, feature_width, "conv");
    c->initial();

    p = new player(nsamples, nfilter, feature_height - filter_height + 1, feature_width - filter_width + 1,
		sampler_height, sampler_width, "pool", 'm');
    p->initial();

    UINT nflat = nfilter * (feature_height - filter_height + 1) * (feature_width - filter_width + 1) / (sampler_height * sampler_width);
    output = new flayer(nsamples, nclass, nflat, true, "output");
    output->initial();

    Matrix *err;
    float ftemp;
    Matrix *deda_mat = new Matrix(nsamples, nflat);
    Matrix *a_mat = new Matrix(nsamples, nflat);
    err = &output->get_deda();
    for(UINT e = 0; e < epochs; ++e) {
        /* main routine */
        cout << "+++++++ epoch " << e + 1 << " +++++++" << endl;

        /* propagate */
        c->propagate(input->get_a());
        p->propagate(c->get_a());
        p->get_a().flatten(*a_mat);
        output->propagate(*a_mat);
        cout << "output is :" << endl;
        output->get_a().display();

        /* get error */
        for(UINT s = 0; s < nsamples; ++s) {
            for(UINT d = 0; d < nclass; ++d) {
                ftemp = label->get_elt(s, d) - output->get_a().get_elt(s, d);
                err->set_elt(s, d, ftemp);
            }
        }

        /* calculate gradient and back propagate */
        output->backprop(*a_mat, *deda_mat);
        p->get_deda().tensorize(*deda_mat);
        p->backprop(c->get_a(), c->get_deda());
        c->backprop(input->get_a(), input->get_deda());

        /* update weights */
        c->updateweights();
        output->updateweights();
    }

    delete a_mat;
    delete deda_mat;
    delete output;
    delete p;
    delete c;
    delete input;
    delete label;
    delete data;

    return 0;
}
