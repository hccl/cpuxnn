#include"dataProcess.h"
#include <stdio.h>
#include <fstream>
#include<assert.h>



DataProcess::DataProcess()
{
	_trainList = "test.scp";
//	_trainList = "C:\\Users\\wzc\\Documents\\Visual Studio 2012\\Projects\\CNNTrain\\CNNTrain\\flayer\\timit_test.scp";
	_trainFileList.clear();
	_mlfFile = " ";
	_maxFrame = 2048;
	_LeftFrmNum = 0;
	_RightFrmNum = 0;
	_ExtentFrm = _LeftFrmNum + _RightFrmNum +1;
	_featDim = 15;
	_minibatch = 5;
	_targetDim = 2;
	_totalFrames = 0;
	_batchNum = 0;
	Featmat=new Matrix(_maxFrame,_featDim);            
	_inputBatch = new Matrix(_minibatch, _featDim*_ExtentFrm);
	_inputLabel = new Matrix(_minibatch, _targetDim);
}

string DataProcess::Trim(const string &source)
{
		size_t beg, end;
		if (source.empty())
			return source;
		beg = source.find_first_not_of(" \a\f\b\r\t\n");
		end = source.find_last_not_of(" \a\f\b\r\t\n");
		if (beg==string::npos)
			return "";
		string temp = source.substr(beg, end-beg+1);
		return temp;
}

vector<string> DataProcess::LoadFeatList()
{
	ifstream fList;
	fList.open(_trainList);
	if (!fList)
	{
		cout<<"Open fileList error!"<<endl;
		exit;
	}	
	string filename;
	while(getline(fList, filename))
	{
		filename = Trim(filename);
		ifstream fFeat(filename);
		if(!fFeat)
		{ 		
			cout<<filename<<" doesn't exists"<<endl;
			continue;
		}
		_trainFileList.push_back(filename);
		
	}
	return _trainFileList;
}

void DataProcess::LoadMLF(const char* mlffile, labmap& label)
{
	ifstream fp(mlffile);
	if(!fp)
	{
		cout << "open " << mlffile << "failed!" <<endl;
		exit(-1);
	}
	string fileLine; 
	string fileName;
	int fbeg,fend,stid;
	vector<int> stateList;
	long tbegin,tend; 
	tbegin = tend  = 0;
	char pureName[256];	
	while(getline(fp,fileLine))
	{
		if (fileLine.empty()|| fileLine[0] == '#')
			continue;
		fileLine = Trim(fileLine);
		if (fileLine[0]=='\"')
		{			
			// get pure filename!
			fileName = fileLine.substr(1,fileLine.find_last_of('.')-1);		;
			strcpy(pureName,fileName.c_str());
			fileName = strlwr(pureName);
			continue;
		}
		if (fileLine == ".")
		{	
			label[fileName] = stateList;
			stateList.clear();
			fileName = "";
			tbegin = tend  = 0;
			continue;
		}
		sscanf(fileLine.c_str(),"%d %d %d",&tbegin,&tend,&stid);
		fbeg = tbegin/FRAMEDUR;fend = tend/FRAMEDUR;
		for (int i=fbeg;i<fend;++i)
			stateList.push_back(stid);		
	}
}

void DataProcess::LoadPLP(const string& file,HtkHead& head,int leftFrm,int rightFrm)
{	
	ifstream fp(file.c_str(),ios::binary);
	if (!fp)
	{
		printf("Load feature file %s failed!\n",file.c_str());
		exit;
	}
	fp.read((char *)&head,sizeof(HtkHeader));
	int extendRow = leftFrm + rightFrm + head.mNSamples;
	int realDim = head.mSampleSize/sizeof(float);
	assert(realDim == _featDim, "feature dim doesn't match: %s", file);
	assert(head.mNSamples <= _maxFrame, "feature file is too big: %s", file);
	fp.read((char *)Featmat->getRowData(0),head.mSampleSize);       
	for (int i=1;i<=leftFrm;++i)
		memcpy(Featmat->getRowData(i),Featmat->getRowData(0),Featmat->width()*sizeof(float));
	int fend = head.mNSamples+leftFrm;
	for (int i=leftFrm+1;i<fend;++i)
		fp.read((char*)Featmat->getRowData(i),head.mSampleSize);		
	for(int i = head.mNSamples+leftFrm;i<extendRow;++i)
		memcpy(Featmat->getRowData(i),Featmat->getRowData(head.mNSamples+leftFrm-1),Featmat->width()*sizeof(float));
	cout<<"Load feature success!"<<endl;
}

void DataProcess::ExpandData(Matrix *input, Matrix *output, int frameIdx)
{
	float* psrc = NULL;
	float* pdes = NULL;
	for (int i = 0; i < _ExtentFrm; i++)
	{
		pdes = output->get_data() + i*_featDim;
		psrc = input->getRowData(frameIdx+i);
		memcpy(pdes,psrc,_featDim*sizeof(float));
	}
}

void DataProcess::MakeBatch(float* inputPtr, float* labPtr, int& frameIndex)
{
	assert(frameIndex < _minibatch, "Push too much frames to minibatch!");
	memcpy(_inputBatch->getRowData(frameIndex),inputPtr,sizeof(float)*_inputBatch->width());
	memcpy(_inputLabel->getRowData(frameIndex),labPtr,sizeof(float)*_targetDim);
	frameIndex++;
	if(_minibatch == frameIndex)
	{
		frameIndex = 0;
		_batchNum+=1;
//		train(_inputBatch, _inputLabel);     /////////////////////////////////////////////////////
	}
}

void DataProcess::LoopFileList(labmap& label)
{
	float* inputVector = new float[_featDim*_ExtentFrm];
	float* outVector = new float[_targetDim];
	int labIndex = 0;
	int frameIndex = 0;
	HtkHeader head;
	_batchNum = 0;
	char tempname[256];
	Matrix* Expandmat = new Matrix(1, _featDim*_ExtentFrm);     
	for(int i=0; i< _trainFileList.size(); i++)
	{					
		string  filename = _trainFileList[i];
		string  purefilename = filename.substr(0,filename.find_last_of('.'));
		strcpy(tempname,filename.c_str());
		purefilename = strlwr(tempname);
		LoadPLP(filename,head,_LeftFrmNum,_RightFrmNum);
		labmap::iterator lab_it = label.find(purefilename);
		if(lab_it == label.end())
		{
			cout << "File: " << filename << "does not have label!!!!!" <<endl;
			continue;
		}
		int nfrmNum = (int)label[purefilename].size();
		if ( nfrmNum != head.mNSamples)
			head.mNSamples = min(nfrmNum,head.mNSamples);
		for(int j = 0; j<head.mNSamples; j++)
		{
			ExpandData(Featmat, Expandmat, j);
			memcpy(inputVector,Expandmat->get_data(),sizeof(float)*_featDim*_ExtentFrm);
			memset(outVector,0,sizeof(float)*_targetDim);
			labIndex = label[purefilename][j];
			outVector[labIndex] = 1.0f;				
			MakeBatch(inputVector, outVector, frameIndex);
		}
		Featmat->zeros_elt();		
	}
	_totalFrames = _batchNum * _minibatch;
	delete Expandmat; Expandmat = NULL;
	delete []inputVector;  inputVector = NULL;
	delete []outVector;	outVector = NULL;
}

DataProcess::~DataProcess()
{
	_trainList = " ";
	_mlfFile = " ";
	_maxFrame = 0;
	_LeftFrmNum = 0;
	_RightFrmNum = 0;
	_ExtentFrm = 0;
	_featDim = 0;
	_totalFrames = 0;
	delete Featmat;
	delete _inputBatch;
	delete _inputLabel;
}

