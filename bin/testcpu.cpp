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

#include <vector>
#include "clayer.h"
#include "flayer.h"
#include "player.h"

using namespace std;
using namespace xnn;

int main(void) {

    Tensor4d *data;
    Matrix *label;
    vector<layer*> network;

    UINT nsamples = 5;
    UINT feature_height = 3;
    UINT feature_width = 5;
    UINT nfilter = 2;
    UINT filter_height = 3;
    UINT filter_width = 2;
    UINT sampler_height = 1;
    UINT sampler_width = 2;
    UINT nclass = 2;
    UINT epochs = 20;

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

    network.push_back(new clayer(nsamples, 1, 1, 1, 1, feature_height, feature_width, "data"));
    network.push_back(new clayer(nsamples, nfilter, filter_height, filter_width, 1, feature_height, feature_width, "conv"));
    network.push_back(new player(nsamples, nfilter, feature_height - filter_height + 1, feature_width - filter_width + 1,
                                 sampler_height, sampler_width, "pool", 'm'));
    UINT nflat = nfilter * (feature_height - filter_height + 1) * (feature_width - filter_width + 1) / (sampler_height * sampler_width);
    network.push_back(new flayer(nsamples, nclass, nflat, true, "output"));

    /* initialize */
    vector<layer*>::iterator lIter;
    vector<layer*>::reverse_iterator rlIter;
    for(lIter = network.begin(); lIter != network.end(); ++lIter) {
        if((*lIter)->get_name() == "data")
            (*lIter)->set_a(*data);
        else
            (*lIter)->initial();
    }

    Matrix *err;
    float ftemp;
    /* intercomm between pooling layer and full layer. It's a mess, to
     * be fixed later. */
    Tensor4d *deda_mat = new Tensor4d(1, 1, nsamples, nflat);
    Tensor4d *a_mat = new Tensor4d(1, 1, nsamples, nflat);
    Tensor4d *deda_lm1;
    const Tensor4d *input;
    err = &((*(network.end()-1))->get_deda().get_kernel(0, 0));
    for(UINT e = 0; e < epochs; ++e) {
        /* main routine */
        cout << "+++++++ epoch " << e + 1 << " +++++++" << endl;

        /* propagate */
        input = &((*(network.begin()))->get_a());
        for(lIter = network.begin(); lIter != network.end(); ++lIter) {
            if((*lIter)->get_name() != "data")
                (*lIter)->propagate(*input);
            if((*lIter)->get_ltype() == layer::ePoolLayer && (*(lIter+1))->get_ltype() == layer::eFullLayer) {
                (*lIter)->get_a().flatten(a_mat->get_kernel(0, 0));
                input = a_mat;
            } else {
                input = &((*lIter)->get_a());
            }
        }
        cout << "output is :" << endl;
        (*(network.end()-1))->get_a().display();

        /* get error */
        for(UINT s = 0; s < nsamples; ++s) {
            for(UINT d = 0; d < nclass; ++d) {
                ftemp = label->get_elt(s, d) - (*(network.end()-1))->get_a().get_elt(0, 0, s, d);
                err->set_elt(s, d, ftemp);
            }
        }

        /* back propagate */
        for(rlIter = network.rbegin(); rlIter != network.rend(); ++rlIter) {
            if((*rlIter)->get_name() != "data") {
                /* calculate gradient and back propagate */
                if((*rlIter)->get_ltype() == layer::eFullLayer && (*(rlIter+1))->get_ltype() == layer::ePoolLayer) {
                    input = a_mat;
                    deda_lm1 = deda_mat;
                } else {
                    input = &((*(rlIter+1))->get_a());
                    deda_lm1 = &((*(rlIter+1))->get_deda());
                }
                if((*rlIter)->get_ltype() == layer::ePoolLayer && (*(rlIter-1))->get_ltype() == layer::eFullLayer)
                    (*rlIter)->get_deda().tensorize(deda_mat->get_kernel(0, 0));
                (*rlIter)->backprop(*input, *deda_lm1);
                /* update weights */
                (*rlIter)->updateweights();
            }
        }
    }

    delete a_mat;
    delete deda_mat;
    delete label;
    delete data;
    while(!network.empty()) {
        delete network.back();
        network.pop_back();
    }
    network.clear();

    return 0;
}
