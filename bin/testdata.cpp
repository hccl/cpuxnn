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
#include "dataProcess.h"
#include "common.h"

using namespace std;
using namespace xnn;

int main(void) {

//    Tensor4d *data;
	Matrix *data;
	Tensor4d* c_data;
//    Matrix *label;
    vector<layer*> network;
/*
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
*/
	UINT nsamples = 5;                //minibatch
	UINT nfeature = 15;               //feature Dim
	UINT nclass = 2;                 //state number
	UINT nnode = 512;                  //hidden units
	UINT epochs = 500;                 //max epoch
	UINT extentFrm = 1;                //extent frames
    UINT feature_height = 3;
    UINT feature_width = 5;
    UINT nfilter = 10;
    UINT filter_height = 3;
    UINT filter_width = 2;
    UINT sampler_height = 1;
    UINT sampler_width = 2;

	bool reshape = 1;
	enumFuncType funcType = TANH;
//	enumFuncType classFunc = SOFTMAX;

    /* network structure */
    assert((feature_height - filter_height + 1) % sampler_height == 0);
    assert((feature_width - filter_width + 1) % sampler_width == 0);

    network.push_back(new clayer(nsamples, 1, 1, 1, 1, feature_height, feature_width, "data"));
    network.push_back(new clayer(nsamples, nfilter, filter_height, filter_width, 1, feature_height, feature_width, "conv"));
    network.push_back(new player(nsamples, nfilter, feature_height - filter_height + 1, feature_width - filter_width + 1,
								sampler_height, sampler_width, "pool", 'm', funcType));
    UINT nflat = nfilter * (feature_height - filter_height + 1) * (feature_width - filter_width + 1) / (sampler_height * sampler_width);
	network.push_back(new flayer(nsamples, nclass, nflat, true, "output", funcType));

    /* initialize */
    vector<layer*>::iterator lIter;
    vector<layer*>::reverse_iterator rlIter;
    for(lIter = network.begin(); lIter != network.end(); ++lIter) {
        if((*lIter)->get_name() == "data")
//            (*lIter)->set_a(*data);
			continue;
        else
            (*lIter)->initial();
    }

	//++++++++++++ loading data ++++++++++++++//
	DataProcess dataProvider;
	labmap label;
	string mlffile = "test.mlf";
//	string mlffile = "C:\\Users\\wzc\\Documents\\Visual Studio 2012\\Projects\\CNNTrain\\CNNTrain\\flayer\\timit_test.mlf";
	dataProvider.LoadMLF(mlffile.c_str(), label);
	vector<string> filelist = dataProvider.LoadFeatList();
	float* inputVector = new float[nfeature*extentFrm];
	float* outVector = new float[nclass];
	int labIndex = 0;
	int frameIndex = 0;
	HtkHeader head;
	UINT _batchNum = 0;
	UINT _totalFrm = 0;
	char tempname[256];
	Matrix* Expandmat = new Matrix(1, nfeature*extentFrm);
	Matrix* Featmat = dataProvider.getFeatmat();
	data = new Matrix(nsamples, nfeature*extentFrm);
	Matrix* labelmat = new Matrix(nsamples, nclass);

    Matrix *err;
    float ftemp;
    /* intercomm between pooling layer and full layer. It's a mess, to
     * be fixed later. */
    Tensor4d *deda_mat = new Tensor4d(1, 1, nsamples, nflat);
    Tensor4d *a_mat = new Tensor4d(1, 1, nsamples, nflat);
    Tensor4d *deda_lm1;
    const Tensor4d *input;
    err = &((*(network.end()-1))->get_deda().get_kernel(0, 0));
	UINT correctFrame = 0;
    for(UINT e = 0; e < epochs; ++e) {
        /* main routine */
        cout << "+++++++ epoch " << e + 1 << " +++++++" << endl;
		for(int i=0; i< filelist.size(); i++)
		{					
			string  filename = filelist[i];
			cout<<"::::"<<filename<<endl;
			string  purefilename = filename.substr(0,filename.find_last_of('.'));
			strcpy(tempname,purefilename.c_str());
			purefilename = strlwr(tempname);
			dataProvider.LoadPLP(filename,head,0,0);
			labmap::iterator lab_it = label.find(purefilename);
			if(lab_it == label.end())
			{
				cout << "File: " << filename << " does not have label!!!!!" <<endl;
				continue;
			}
			int nfrmNum = (int)label[purefilename].size();
			if ( nfrmNum != head.mNSamples)
				head.mNSamples = min(nfrmNum,head.mNSamples);
			for(int j = 0; j<head.mNSamples; j++)
			{
				dataProvider.ExpandData(Featmat, Expandmat, j);
				memcpy(inputVector,Expandmat->get_data(),sizeof(float)*nfeature*extentFrm);
				memset(outVector,0,sizeof(float)*nclass);
				labIndex = label[purefilename][j];
				outVector[labIndex] = 1.0f;				
				memcpy(data->getRowData(frameIndex),inputVector,sizeof(float)*data->width());
				memcpy(labelmat->getRowData(frameIndex),outVector,sizeof(float)*nclass);
				frameIndex++;
				if(nsamples == frameIndex)
				{
					frameIndex = 0;
					_batchNum+=1;
					if ( reshape )
					{
						c_data = new Tensor4d(nsamples, extentFrm, feature_height, feature_width);
						c_data->tensorize(*data);
					}
					 /* propagate */
					(*(network.begin()))->set_a(*c_data);
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
						float maxprob = 0.0f;
						UINT probIdx = 0;
						for(UINT d = 0; d < nclass; ++d) {
							ftemp = labelmat->get_elt(s, d) - (*(network.end()-1))->get_a().get_elt(0, 0, s, d);
							err->set_elt(s, d, ftemp);
							if (maxprob < (*(network.end()-1))->get_a().get_elt(0, 0, s, d))
							{
								maxprob = (*(network.end()-1))->get_a().get_elt(0, 0, s, d);					
								probIdx = d;
							}
						}
						if(labelmat->get_elt(s,probIdx) == 1){
							correctFrame+=1;							
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
			}
			Featmat->zeros_elt();
		}
		_totalFrm = _batchNum * nsamples;
		cout<<"Frame accuracy: "<<correctFrame<<"/"<<_totalFrm<<"="<<(float)correctFrame/_totalFrm<<endl;
		correctFrame = 0;
		_batchNum = 0;
		frameIndex = 0;
		cout << "------- epoch " << e + 1 << " finished -------" << endl;
	}

    delete a_mat;
    delete deda_mat;
    delete labelmat;
    delete data;
	delete c_data;
	delete Expandmat; Expandmat = NULL;
	delete []inputVector;  inputVector = NULL;
	delete []outVector;	outVector = NULL;
    while(!network.empty()) {
        delete network.back();
        network.pop_back();
    }
    network.clear();

    return 0;
}
