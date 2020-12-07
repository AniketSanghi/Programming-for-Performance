#!/bin/bash

g++ -O2 -o problem2 problem2.cpp
icc -O2 -o problem2_icc problem2.cpp

echo "GCC"
./problem2

echo "ICC"
./problem2_icc
