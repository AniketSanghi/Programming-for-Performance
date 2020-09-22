#!/bin/bash

g++ a.cpp -o a
g++ b.cpp -o b

time ./a
time ./b

time ./a
time ./b

rm a b