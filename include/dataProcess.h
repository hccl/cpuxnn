
#ifndef DATAPROCESS_H
#define DATAPROCESS_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <stdio.h>
#include <map>
#include "matrix.h"
#include "common.h"

using namespace std;
using namespace xnn;
//typedef map<string,vector<int> > labmap;
#define FRAMEDUR 100000;

typedef struct HtkHead
{
    int    mNSamples;              
    int    mSamplePeriod;
    short int    mSampleSize;
    unsigned short int   mSampleKind;
}HtkHeader;

class DataProcess
{
public:
    DataProcess();
    ~DataProcess();
    vector<string> LoadFeatList();
    void LoadMLF(const char* mlffile, labmap &label);
    void LoadPLP(const string& file,HtkHead& head,int leftFrm,int rightFrm);
    void ExpandData(Matrix* input, Matrix* output, int frameIdx);
    void LoopFileList(labmap& label);
    void MakeBatch(float* inputPtr, float* labPtr, int& frameIndex);
    Matrix* getFeatmat() {return Featmat;}; 
    string Trim(const string &source);

private:
    int _LeftFrmNum;
    int _RightFrmNum;
    int _maxFrame;
    int _ExtentFrm;
    int _featDim;
    int _minibatch;
    int _targetDim;  //state num
    int _batchNum;
    unsigned long _totalFrames; 
    Matrix* Featmat;
    Matrix* _inputBatch;
    Matrix* _inputLabel;
    string _trainList;
    string _mlfFile;
    vector<string> _trainFileList;    
};

#endif

