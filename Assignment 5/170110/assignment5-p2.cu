// Compile: nvcc -g -G -arch=sm_61 -std=c++11 assignment5-p2.cu -o assignment5-p2

#include <algorithm>
#include <cuda.h>
#include <iostream>
#include <sys/time.h>

#define THRESHOLD (0.000001)
#define EPS_THREADS_PER_BLOCK 64
#define EPS_BLOCK_SIZE 4096

#define SUM_THREADS_PER_BLOCK 64
#define SUM_BLOCK_SIZE 8

using std::cerr;
using std::cout;
using std::endl;

// Final Time:
// Serial time on CPU: 47.7951 msec
// Kernel time on GPU: 30.0346 msec

__host__ void host_excl_prefix_sum(float* h_A, float* h_O, int N) {
  h_O[0] = 0;
  for (int i = 1; i < N; i++) {
    h_O[i] = h_O[i - 1] + h_A[i - 1];
  }
}

__global__ void kernel_excl_prefix_sum(float* d_in, float* d_out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  float prev = 0.0;
  for(int k = i * EPS_BLOCK_SIZE + 1; k < (i+1)*EPS_BLOCK_SIZE; k++) {
    prev = prev + d_in[k-1];
    d_out[k] = prev;
  }
}

__global__ void kernel_sum(float *d_in, float *d_out, int block_size) {
  int gi = (blockIdx.x * blockDim.x + threadIdx.x)*SUM_BLOCK_SIZE;

  int startIndex = (2 * (gi / block_size) + 1)*block_size;
  int i = startIndex + (gi % block_size);

  __shared__ float toAdd;

  if(threadIdx.x == 0) {
    toAdd = d_out[startIndex - 1] + d_in[startIndex - 1];
  }
  __syncthreads();

  int cnt = i + SUM_BLOCK_SIZE;
  for(; i < cnt; ++i)
    d_out[i] = d_out[i] + toAdd;
}



// __global__ void kernel_sum(float *d_in, float *d_out, int block_size) {
//   int gi = (blockIdx.x * blockDim.x + threadIdx.x);

//   int startIndex = (2 * (gi / block_size) + 1)*block_size;
//   int i = startIndex + (gi % block_size);

//   __shared__ float toAdd;

//   if(threadIdx.x == 0) {
//     toAdd = d_out[startIndex - 1] + d_in[startIndex - 1];
//   }
//   __syncthreads();

//     d_out[i] = d_out[i] + toAdd;
// }



// __global__ void kernel_excl_prefix_sum(float* d_in, float* d_out) {
//   int i = blockIdx.y * blockDim.y + threadIdx.y;
//   int j = blockIdx.x * blockDim.x + threadIdx.x;

//   if(i > j) {
//     atomicAdd(&d_out[i], d_in[j]);
//   }
// }

__host__ void check_result(float* w_ref, float* w_opt, int N) {
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
    cout << numdiffs << " Diffs found over threshold " << THRESHOLD << "; Max Diff = " << maxdiff
         << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}

__host__ double rtclock() { // Seconds
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0) {
    cout << "Error return from gettimeofday: " << stat << "\n";
  }
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void safe(cudaError_t status) {
  if (status != cudaSuccess) {
    cerr << cudaGetErrorString(status) << endl;
  }
}

int main() {
  const int N = (1 << 24);
  size_t size = N * sizeof(float);

  float* h_in = (float*)malloc(size);
  std::fill_n(h_in, N, 1);

  float* h_excl_sum_out = (float*)malloc(size);
  std::fill_n(h_excl_sum_out, N, 0);

  double clkbegin = rtclock();
  host_excl_prefix_sum(h_in, h_excl_sum_out, N);
  double clkend = rtclock();
  double time = clkend - clkbegin; // seconds
  cout << "Serial time on CPU: " << time * 1000 << " msec" << endl;

  float* h_dev_result = (float*)malloc(size);
  std::fill_n(h_dev_result, N, 0);
  float* d_in;
  float* d_out;
  cudaError_t status;
  cudaEvent_t start, end;
  
  safe(cudaEventCreate(&start));
  safe(cudaEventCreate(&end));

  safe(status = cudaMalloc(&d_in, size));
  safe(status = cudaMalloc(&d_out, size));

  safe(cudaEventRecord(start,0));

  // --- START --- //
  safe(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));
  safe(cudaMemcpy(d_out, h_dev_result, size, cudaMemcpyHostToDevice));

  dim3 threadsPerBlock(EPS_THREADS_PER_BLOCK);
  dim3 numBlocks(N / (threadsPerBlock.x * EPS_BLOCK_SIZE));
  kernel_excl_prefix_sum<<<numBlocks, threadsPerBlock>>>(d_in, d_out);

  int numberOfThreads = N / (2 * SUM_BLOCK_SIZE);
  dim3 sumThreadsPerBlock(SUM_THREADS_PER_BLOCK);
  dim3 sumNumBlocks(numberOfThreads / sumThreadsPerBlock.x);

  for(int iter = EPS_BLOCK_SIZE; iter < N; iter *= 2) {
    kernel_sum<<<sumNumBlocks, sumThreadsPerBlock>>>(d_in, d_out, iter);
  }

  safe(cudaMemcpy(h_dev_result, d_out, size, cudaMemcpyDeviceToHost));
  // --- END --- //

  float k_time;
  safe(cudaEventRecord(end,0));
  safe(cudaEventSynchronize(end));
  safe(cudaEventElapsedTime(&k_time,start,end));

  check_result(h_excl_sum_out, h_dev_result, N);
  cout << "Kernel time on GPU: " << k_time << " msec" << endl;

  safe(cudaEventDestroy(start));
  safe(cudaEventDestroy(end));

  if (status != cudaSuccess) {
    cerr << cudaGetErrorString(status) << endl;
  }

  // Free device memory
  safe(cudaFree(d_in));
  safe(cudaFree(d_out));

  free(h_in);
  free(h_excl_sum_out);
  free(h_dev_result);

  return EXIT_SUCCESS;
}
