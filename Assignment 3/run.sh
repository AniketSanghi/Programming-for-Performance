#!/bin/bash

g++ -O2 -march=native -o problem1 problem1.cpp
icc -O2 -o problem1_icc problem1.cpp

echo "GCC"
./problem1

echo "ICC"
./problem1_icc
