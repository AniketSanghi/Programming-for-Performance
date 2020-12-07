// Compile: nvcc -g -G -arch=sm_61 hello-world.cu -o hello-world

#include <cstdio>
#include <cuda.h>
#include <iostream>

__global__ void hwkernel() { printf("Hello World!\n"); }

int main() {
  std::cout << "Before launching kernel" << std::endl << std::endl;

  hwkernel<<<1, 8>>>();
  cudaDeviceSynchronize();

  return EXIT_SUCCESS;
}
