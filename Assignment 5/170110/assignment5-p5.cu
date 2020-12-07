// Compile: nvcc -g -G -arch=sm_61 -std=c++11 assignment5-p5.cu -o assignment5-p5

#include <cmath>
#include <cstdlib>
#include <cuda.h>
#include <iostream>
#include <sys/time.h>

const uint64_t N = (512);
#define THRESHOLD (0.000001)
#define BLOCK_SIZE 8

using std::cerr;
using std::cout;
using std::endl;


__global__ void kernel1(float *in, float *out) {
  int i = blockIdx.z * blockDim.z + threadIdx.z;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.x * blockDim.x + threadIdx.x;

  if(i > 0 && i < N-1 && j > 0 && j < N-1 && k > 0 && k < N-1) {
    out[i*N*N + j*N + k] = 0.8 * (in[(i-1)*N*N + j*N + k] + in[(i+1)*N*N + j*N + k] + in[i*N*N + (j-1)*N + k] 
                             + in[i*N*N + (j+1)*N + k] + in[i*N*N + j*N + k-1] + in[i*N*N + j*N + k+1]);
  }
}

// N = 64, BLOCK_SIZE = 8
// Stencil time on CPU: 2.92015 msec
// Kernel 1 time (ms): 3.4337
// Kernel 2 time (ms): 1.69574
// ---------------------------------
// Stencil time on CPU: 2.19202 msec
// Kernel 1 time (ms): 1.50995
// Kernel 2 time (ms): 1.53014


// N = 512, BLOCK_SIZE = 4
// Stencil time on CPU: 1046.3 msec
// Kernel 1 time (ms): 637.759
// Kernel 2 time (ms): 676.278
// ----------------------------------
// N = 512, BLOCK_SIZE = 8
// Stencil time on CPU: 1056.64 msec
// Kernel 1 time (ms): 655.629
// Kernel 2 time (ms): 658.527
__global__ void kernel2(float *in, float *out) {
  int i = threadIdx.z;
  int j = threadIdx.y;
  int k = threadIdx.x;

  int gi = (blockIdx.z * BLOCK_SIZE + i);
  int gj = (blockIdx.y * BLOCK_SIZE + j);
  int gk = (blockIdx.x * BLOCK_SIZE + k);

  __shared__ float A[BLOCK_SIZE][BLOCK_SIZE][BLOCK_SIZE];

  A[i][j][k] = in[gi*N*N + gj*N + gk];

  __syncthreads();

  if (k > 0 && k < BLOCK_SIZE - 1 && j > 0 && j < BLOCK_SIZE - 1 && i > 0 && i < BLOCK_SIZE - 1) {
    out[gi*N*N + gj*N + gk] = 0.8 * (A[i-1][j][k] + A[i+1][j][k] + A[i][j-1][k]
                                                  + A[i][j+1][k] + A[i][j][k-1] + A[i][j][k+1]);
  } else {
    if(gi > 0 && gi < N - 1 && gj > 0 && gj < N - 1 && gk > 0 && gk < N - 1) {
      out[gi*N*N + gj*N + gk] = 0.8 * (in[(gi-1)*N*N + gj*N + gk] 
                                  + in[(gi+1)*N*N + gj*N + gk] + in[gi*N*N + (gj-1)*N + gk] 
                                  + in[gi*N*N + (gj+1)*N + gk] + in[gi*N*N + gj*N + gk-1] 
                                  + in[gi*N*N + gj*N + gk+1]);
    }
  }
}

__host__ void stencil(float *in, float *out) {
  for(uint64_t i = 1; i < N - 1; i++) {
    for(uint64_t j = 1; j < N - 1; j++) {
      for(uint64_t k = 1; k < N - 1; k++) {
        out[i*N*N + j*N + k] = 0.8 * (in[(i-1)*N*N + j*N + k] + in[(i+1)*N*N + j*N + k] + in[i*N*N + (j-1)*N + k] 
                             + in[i*N*N + (j+1)*N + k] + in[i*N*N + j*N + k-1] + in[i*N*N + j*N + k+1]);
      }
    }
  }
}

__host__ void check_result(float* w_ref, float* w_opt, uint64_t size) {
  double maxdiff = 0.0, this_diff = 0.0;
  int numdiffs = 0;

  for (uint64_t i = 0; i < size; i++) {
    for (uint64_t j = 0; j < size; j++) {
      for (uint64_t k = 0; k < size; k++) {
        this_diff = w_ref[i + N * j + N * N * k] - w_opt[i + N * j + N * N * k];
        if (std::fabs(this_diff) > THRESHOLD) {
          numdiffs++;
          if (this_diff > maxdiff) {
            maxdiff = this_diff;
          }
        }
      }
    }
  }

  if (numdiffs > 0) {
    cout << numdiffs << " Diffs found over THRESHOLD " << THRESHOLD << "; Max Diff = " << maxdiff
         << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}


double rtclock() { // Seconds
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
  uint64_t SIZE = N * N * N;

  float *h_in, *h_out, *h_gpu_out1, *h_gpu_out2;

  h_in = (float*)malloc(SIZE * sizeof(float));
  h_out = (float*)malloc(SIZE * sizeof(float));
  h_gpu_out1 = (float*)malloc(SIZE * sizeof(float));
  h_gpu_out2 = (float*)malloc(SIZE * sizeof(float));

  for(uint64_t i = 0; i < SIZE; ++i) {
    h_in[i] = rand() % 100000000;
    h_out[i] = 0;
    h_gpu_out1[i] = 0;
    h_gpu_out2[i] = 0;
  }

  double clkbegin = rtclock();
  stencil(h_in, h_out);
  double clkend = rtclock();
  double cpu_time = clkend - clkbegin;
  cout << "Stencil time on CPU: " << cpu_time * 1000 << " msec" << endl;

  // cudaError_t status;
  cudaEvent_t start, end;
  float *d_in, *d_out1, *d_out2;

  safe(cudaMalloc(&d_in, SIZE * sizeof(float)));
  safe(cudaMalloc(&d_out1, SIZE * sizeof(float)));

  safe(cudaEventCreate(&start));
  safe(cudaEventCreate(&end));

  safe(cudaEventRecord(start, 0));

  // --- START --- //
  safe(cudaMemcpy(d_in, h_in, SIZE * sizeof(float), cudaMemcpyHostToDevice));
  safe(cudaMemcpy(d_out1, h_gpu_out1, SIZE * sizeof(float), cudaMemcpyHostToDevice));

  dim3 threadsPerBlock(8, 8, 8);
  dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y, (N + threadsPerBlock.z - 1) / threadsPerBlock.z);
  kernel1<<<numBlocks, threadsPerBlock>>>(d_in, d_out1);

  safe(cudaMemcpy(h_gpu_out1, d_out1, SIZE * sizeof(float), cudaMemcpyDeviceToHost));
  // --- END --- //

  safe(cudaEventRecord(end, 0));
  safe(cudaEventSynchronize(end));

  check_result(h_out, h_gpu_out1, N);
  float kernel_time;
  cudaEventElapsedTime(&kernel_time, start, end);
  std::cout << "Kernel 1 time (ms): " << kernel_time << "\n";

  safe(cudaEventDestroy(start));
  safe(cudaEventDestroy(end));


  safe(cudaMalloc(&d_out2, SIZE * sizeof(float)));

  safe(cudaEventCreate(&start));
  safe(cudaEventCreate(&end));

  safe(cudaEventRecord(start, 0));

  // --- START --- //
  safe(cudaMemcpy(d_out2, h_gpu_out2, SIZE * sizeof(float), cudaMemcpyHostToDevice));

  dim3 threadsPerBlock2(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
  dim3 numBlocks2((N + threadsPerBlock2.x - 1) / threadsPerBlock2.x, (N + threadsPerBlock2.y - 1) / threadsPerBlock2.y, (N + threadsPerBlock2.z - 1) / threadsPerBlock2.z);
  kernel2<<<numBlocks2, threadsPerBlock2>>>(d_in, d_out2);

  safe(cudaMemcpy(h_gpu_out2, d_out2, SIZE * sizeof(float), cudaMemcpyDeviceToHost));
  // --- END --- //

  safe(cudaEventRecord(end, 0));
  safe(cudaEventSynchronize(end));


  check_result(h_out, h_gpu_out2, N);
  cudaEventElapsedTime(&kernel_time, start, end);
  std::cout << "Kernel 2 time (ms): " << kernel_time << "\n";

  safe(cudaEventDestroy(start));
  safe(cudaEventDestroy(end));

  safe(cudaFree(d_in));
  safe(cudaFree(d_out1));
  safe(cudaFree(d_out2));

  free(h_in);
  free(h_out);
  free(h_gpu_out1);
  free(h_gpu_out2);

  return EXIT_SUCCESS;
}
