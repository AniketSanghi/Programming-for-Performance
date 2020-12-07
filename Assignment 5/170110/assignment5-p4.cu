// Compile: nvcc -g -G -arch=sm_61 -std=c++11 assignment5-p4.cu -o assignment5-p4

#include <cmath>
#include <cuda.h>
#include <iostream>
#include <sys/time.h>

const uint64_t N = (1 << 10);
#define THRESHOLD (0.000001)

#define BLOCK_SIZE 32

using std::cerr;
using std::cout;
using std::endl;


// BLOCK_SIZE 8
// N = 4096
// Matmul time on CPU: 986951 msec
// Kernel 1 time (ms): 56860.2
// Kernel 2 time (ms): 5740.13
// N = 2048
// Matmul time on CPU: 121404 msec
// Kernel 1 time (ms): 4763.18
// Kernel 2 time (ms): 671.315
// N = 1024
// Matmul time on CPU: 4147.87 msec
// Kernel 1 time (ms): 620.725
// Kernel 2 time (ms): 75.2633


// BLOCK_SIZE 16
// N = 4096
// Matmul time on CPU: 986951 msec
// Kernel 1 time (ms): 31560.8
// Kernel 2 time (ms): 4516.1
// N = 2048
// Matmul time on CPU: 121404 msec
// Kernel 1 time (ms): 4014.84
// Kernel 2 time (ms): 567.335
// N = 1024
// Matmul time on CPU: 4147.87 msec
// Kernel 1 time (ms): 554.005
// Kernel 2 time (ms): 73.2247


// BLOCK_SIZE 32
// N = 4096
// Matmul time on CPU: 985104 msec
// Kernel 1 time (ms): 31512
// Kernel 2 time (ms): 4127.28
// N = 2048
// Matmul time on CPU: 120785 msec
// Kernel 1 time (ms): 4014.84
// Kernel 2 time (ms): 523.793
// N = 1024
// Matmul time on CPU: 4235.79 msec
// Kernel 1 time (ms): 554.005
// Kernel 2 time (ms): 69.2259


// BLOCK_SIZE 32 with unroll
// N = 4096
// Matmul time on CPU: 976394 msec
// Kernel 1 time (ms): 39812.4
// Kernel 2 time (ms): 3617.57
// N = 2048
// Matmul time on CPU: 112694 msec
// Kernel 1 time (ms): 4014.84
// Kernel 2 time (ms): 435.474
// N = 1024
// Matmul time on CPU: 4378.58 msec
// Kernel 1 time (ms): 554.005
// Kernel 2 time (ms): 66.154



// ---------------------------------------------------
__global__ void kernel1(uint64_t* d_A, uint64_t* d_B, uint64_t* d_C) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  uint64_t val = 0;
  for(uint64_t k = 0; k < N; k++) {
    val += d_A[i * N + k] * d_B[k * N + j];
  }
  d_C[i * N + j] = val;
}

__global__ void kernel2(uint64_t* d_A, uint64_t* d_B, uint64_t* d_C) {
  int blocki = blockIdx.y * blockDim.y;
  int blockj = blockIdx.x * blockDim.x;

  int i = threadIdx.y;
  int j = threadIdx.x;

  uint64_t val = 0;

  for(int k = 0; k < N / BLOCK_SIZE; k++) {

    // Allocate shared memory for block of threads
    __shared__ double Asub[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double Bsub[BLOCK_SIZE][BLOCK_SIZE];

    // Each thread in block bringing one element from banks
    Asub[i][j] = d_A[(blocki + i)*N + (k * BLOCK_SIZE + j)];
    Bsub[i][j] = d_B[(k * BLOCK_SIZE + i)*N + (blockj + j)];

    // To ensure all element loaded
    __syncthreads();

    for(int k2 = 0; k2 < BLOCK_SIZE; k2+=8) {
      val = val + Asub[i][k2] * Bsub[k2][j]
                + Asub[i][k2+1] * Bsub[k2+1][j]
                + Asub[i][k2+2] * Bsub[k2+2][j]
                + Asub[i][k2+3] * Bsub[k2+3][j]
                + Asub[i][k2+4] * Bsub[k2+4][j]
                + Asub[i][k2+5] * Bsub[k2+5][j]
                + Asub[i][k2+6] * Bsub[k2+6][j]
                + Asub[i][k2+7] * Bsub[k2+7][j];
    }

    // Make sure all computations completed
    __syncthreads();
  }

  d_C[(blocki + i) * N + (blockj + j)] = val;  
}

__host__ void cpumatMul(uint64_t* h_A, uint64_t* h_B, uint64_t* h_C) {
  for (uint64_t i = 0; i < N; i++) {
    for (uint64_t j = 0; j < N; j++) {
      uint64_t sum = 0;
      for (uint64_t k = 0; k < N; k++) {
        sum += h_A[i * N + k] * h_B[k * N + j];
      }
      h_C[i * N + j] = sum;
    }
  }
}

__host__ void check_result(uint64_t* w_ref, uint64_t* w_opt) {
  bool wrong = false;
  for (uint64_t i = 0; i < N; i++) {
    for (uint64_t j = 0; j < N; j++) {
      if (w_ref[i * N + j] != w_opt[i * N + j]) {
        wrong = true;
        goto out;
      }
    }
  }
out:
  if (wrong) {
    cout << " Diffs found!" << endl;
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
  uint64_t SIZE = N * N;

  uint64_t *h_A, *h_B, *h_cpu_C, *h_gpu1_C, *h_gpu2_C;

  h_A = (uint64_t*)malloc(SIZE * sizeof(uint64_t));
  h_B = (uint64_t*)malloc(SIZE * sizeof(uint64_t));
  h_cpu_C = (uint64_t*)malloc(SIZE * sizeof(uint64_t));
  h_gpu1_C = (uint64_t*)malloc(SIZE * sizeof(uint64_t));
  h_gpu2_C = (uint64_t*)malloc(SIZE * sizeof(uint64_t));

  for (uint64_t i = 0; i < N; i++) {
    for (uint64_t j = 0; j < N; j++) {
      h_A[i * N + j] = rand() % 100000;
      h_B[i * N + j] = rand() % 100000;
      h_cpu_C[i * N + j] = 0;
      h_gpu1_C[i * N + j] = 0;
      h_gpu2_C[i * N + j] = 0;
    }
  }

  double clkbegin = rtclock();
  cpumatMul(h_A, h_B, h_cpu_C);
  double clkend = rtclock();
  double cpu_time = clkend - clkbegin;
  cout << "Matmul time on CPU: " << cpu_time * 1000 << " msec" << endl;

  cudaError_t status;
  cudaEvent_t start, end;

  uint64_t *d_A, *d_B, *d_C1;
  status = cudaMalloc(&d_A, SIZE * sizeof(uint64_t));
  if (status != cudaSuccess) {
    cerr << cudaGetErrorString(status) << endl;
  }
  safe(cudaMalloc(&d_B, SIZE * sizeof(uint64_t)));
  safe(cudaMalloc(&d_C1, SIZE * sizeof(uint64_t)));

  safe(cudaEventCreate(&start));
  safe(cudaEventCreate(&end));

  safe(cudaEventRecord(start, 0));

  // --- START --- //
  safe(cudaMemcpy(d_A, h_A, SIZE * sizeof(uint64_t), cudaMemcpyHostToDevice));
  safe(cudaMemcpy(d_B, h_B, SIZE * sizeof(uint64_t), cudaMemcpyHostToDevice));

  dim3 threadsPerBlock(32, 32);
  dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
  kernel1<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C1);

  safe(cudaMemcpy(h_gpu1_C, d_C1, SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost));
  // --- END --- //

  safe(cudaEventRecord(end, 0));
  safe(cudaEventSynchronize(end));

  check_result(h_cpu_C, h_gpu1_C);
  float kernel_time;
  cudaEventElapsedTime(&kernel_time, start, end);
  std::cout << "Kernel 1 time (ms): " << kernel_time << "\n";

  safe(cudaEventDestroy(start));
  safe(cudaEventDestroy(end));

  uint64_t* d_C2;
  safe(cudaMalloc(&d_C2, SIZE * sizeof(uint64_t)));

  safe(cudaEventCreate(&start));
  safe(cudaEventCreate(&end));

  safe(cudaEventRecord(start, 0));

  // --- START --- //
  safe(cudaMemcpy(d_A, h_A, SIZE * sizeof(uint64_t), cudaMemcpyHostToDevice));
  safe(cudaMemcpy(d_B, h_B, SIZE * sizeof(uint64_t), cudaMemcpyHostToDevice));

  dim3 threadsPerBlock2(BLOCK_SIZE, BLOCK_SIZE);
  dim3 numBlocks2((N + threadsPerBlock2.x - 1) / threadsPerBlock2.x, (N + threadsPerBlock2.y - 1) / threadsPerBlock2.y);
  kernel2<<<numBlocks2, threadsPerBlock2>>>(d_A, d_B, d_C2);
  
  safe(cudaMemcpy(h_gpu2_C, d_C2, SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost));
  // --- END ---- //

  safe(cudaEventRecord(end, 0));
  safe(cudaEventSynchronize(end));

  check_result(h_cpu_C, h_gpu2_C);
  cudaEventElapsedTime(&kernel_time, start, end);
  std::cout << "Kernel 2 time (ms): " << kernel_time << "\n";

  safe(cudaEventDestroy(start));
  safe(cudaEventDestroy(end));

  safe(cudaFree(d_A));
  safe(cudaFree(d_B));
  safe(cudaFree(d_C1));
  safe(cudaFree(d_C2));

  free(h_A);
  free(h_B);
  free(h_cpu_C);
  free(h_gpu1_C);
  free(h_gpu2_C);

  return EXIT_SUCCESS;
}
