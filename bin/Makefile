#
# Copyright 2014 by Institute of Acoustics, Chinese Academy of Sciences
# http://www.ioa.ac.cn
#
# See the file COPYING for the licence associated with this software.
#
# Author(s):
#   Xingyu Na, August 2014
#

SHELL = /bin/sh
CXX = g++
CXXFLAGS = -g -I../include
XNNLIB = ../src/cpuxnn.a

PROGS = testcpu

all: $(PROGS)

%: %.cpp $(XNNLIB)
	$(CXX) $(CXXFLAGS) -o $@ $^
