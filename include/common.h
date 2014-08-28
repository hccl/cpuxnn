/*
 * Copyright 2014 by Institute of Acoustics, Chinese Academy of Sciences
 * http://www.ioa.ac.cn
 *
 * See the file COPYING for the licence associated with this software.
 *
 * Author(s):
 *   Xingyu Na, August 2014
 */

#ifndef COMMON_H_
#define COMMON_H_

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <exception>
#include <stdexcept>
#include <vector>
#include <list>
#include <cstddef>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cassert>
#include <map>
#include <cstdio>
using namespace std;

typedef unsigned int UINT;
typedef unsigned short USHORT;
typedef map<string,vector<int> > labmap;
typedef enum 
{
	SIGMOID,
	TANH,
	LINEAR,
	RELU
}enumFuncType;

#define FMAX_VALUE 1e+10;
#define FMIN_VALUE -1e+10;

#endif /* COMMON_H_ */
