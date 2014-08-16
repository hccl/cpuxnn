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
srcdir = .
prjdir = ..
prefix = ../lib

CXX    = g++
CXXFLAGS = -c -I../include
RANLIB = ranlib
CFLAGS = -I../include
target = cpuxnn.a
objects = matrix.o \
	tensor.o \
	flayer.o \
	clayer.o \
	player.o

all: cpuxnn.a

cpuxnn.a: $(objects)
	ar rv $@ $^
	$(RANLIB) $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

.PHONY: clean

clean:
	rm -f $(objects) cpuxnn.a
