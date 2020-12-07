// Compile: nvcc -g -G -arch=sm_61 vec-add-events.cu -o vec-add-events

#include <cstddef>
#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <stdio.h>

using std::cout;
using std::endl;

const double THRESHOLD = 0.0000001;

__global__ void dkernel(float *a, float *b, float *c, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    c[idx] = a[idx] + b[idx];
  }
}

void hkernel(float *a, float *b, float *c, size_t size) {
  for (int i = 0; i < size; i++) {
    c[i] = a[i] + b[i];
  }
}

__host__ void check_result(float *w_ref, float *w_opt, int N) {
  double maxdiff = 0.0, this_diff = 0.0;
  int numdiffs = 0;

  for (int i = 0; i < N; i++) {
    this_diff = w_ref[i] - w_opt[i];
    if (fabs(this_diff) > THRESHOLD) {
      numdiffs++;
      if (this_diff > maxdiff)
        maxdiff = this_diff;
    }
  }

  if (numdiffs > 0) {
    cout << numdiffs << " Diffs found over threshold " << THRESHOLD
         << "; Max Diff = " << maxdiff << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t vecAdd_wrapper(float *h_a, float *h_b, float *h_c, size_t num) {
  float *dev_a = 0;
  float *dev_b = 0;
  float *dev_c = 0;

  size_t size = num * sizeof(float);
  cudaError_t cudaStatus;
  cudaEvent_t start, stop;
  int threadsPerBlock = 512;
  int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;

  // Set device to use
  cudaStatus = cudaSetDevice(0);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaSetDevice() failed");
    goto Error;
  }

  cudaStatus = cudaMalloc(&dev_a, size);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc() failed");
    goto Error;
  }
  cudaStatus = cudaMalloc(&dev_b, size);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc() failed");
    goto Error;
  }
  cudaStatus = cudaMalloc(&dev_c, size);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc() failed");
    goto Error;
  }

  cudaStatus = cudaMemcpy(dev_a, h_a, size, cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy() failed");
    goto Error;
  }
  cudaStatus = cudaMemcpy(dev_b, h_b, size, cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy() failed");
    goto Error;
  }

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);

  dkernel<<<numBlocks, threadsPerBlock>>>(dev_a, dev_b, dev_c, num);

  // cudaDeviceSynchronize waits for the cuda kernel to finish
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr,
            "cudaDeviceSynchronize() returned error code %d in dkernel "
            "execution\n",
            cudaStatus);
    goto Error;
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  // gpu_time will contain vector addition time at gpu without copying overhead
  float gpu_time;
  cudaEventElapsedTime(&gpu_time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  printf("Kernel time (ms) without copy overhead: %lf\n", gpu_time);

  cudaStatus = cudaMemcpy(h_c, dev_c, size, cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy() failed!");
  }

Error:
  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);

  return cudaStatus;
}

int main() {
  const int N = (1 << 24);
  float gpu_time;
  float cpu_time;

  float *h_a = (float *)malloc(N * sizeof(float));
  float *h_b = (float *)malloc(N * sizeof(float));
  float *h_gpu = (float *)malloc(N * sizeof(float));
  float *h_cpu = (float *)malloc(N * sizeof(float));

  int value_1 = 1;
  int value_2 = 2;

  std::fill_n(h_a, N, value_1);
  std::fill_n(h_b, N, value_2);
  std::fill_n(h_gpu, N, 0);
  std::fill_n(h_cpu, N, 0);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
  // Add in parallel from helper function with cuda_helper cuda kernel
  cudaError_t cudaStatus = vecAdd_wrapper(h_a, h_b, h_gpu, N);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "add with cuda failed!");
    return EXIT_FAILURE;
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  // gpu_time will contain vector addition time at gpu with copying overhead
  cudaEventElapsedTime(&gpu_time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  printf("Kernel time (ms) with copy overhead: %lf\n", gpu_time);
  cudaStatus = cudaThreadExit();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaThreadExit() failed!");
    return EXIT_FAILURE;
  }

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
  hkernel(h_a, h_b, h_cpu, N);
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);

  cudaEventElapsedTime(&cpu_time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  printf("CPU time (ms): %lf\n", cpu_time);

  check_result(h_gpu, h_cpu, N);

  free(h_a);
  free(h_b);
  free(h_cpu);
  free(h_gpu);

  return EXIT_SUCCESS;
}
