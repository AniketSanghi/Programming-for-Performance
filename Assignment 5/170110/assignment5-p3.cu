// Compile: nvcc -g -G -arch=sm_61 -std=c++11 assignment5-p3.cu -o assignment5-p3

#include <cmath>
#include <iostream>
#include <sys/time.h>

#define SIZE 4096
#define THRESHOLD (0.000001)
#define BLOCK_SIZE 32

using std::cerr;
using std::cout;
using std::endl;

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

__host__ void ATAonCPU(double* M, double* P) {
  for (int k = 0; k < SIZE; k++) {
    for (int i = 0; i < SIZE; i++) {
      for (int j = 0; j < SIZE; j++)
        P[i*SIZE + j] += M[k*SIZE + i] * M[k*SIZE + j];
    }
  }
}

__host__ void check_result(double* Test, double* Ref) {
  double maxdiff = 0, rel_diff = 0;
  int numdiffs = 0;

  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      rel_diff = (Test[i*SIZE + j] - Ref[i*SIZE + j]);
      if (fabs(rel_diff) > THRESHOLD) {
        numdiffs++;
        if (rel_diff > maxdiff)
          maxdiff = rel_diff;
      }
    }
  }
  if (numdiffs > 0)
    cout << numdiffs << " Diffs found over THRESHOLD " << THRESHOLD << " Max Diff = " << maxdiff
         << "\n";
  else
    cout << "No differences found between base and test versions\n";
}


// -----------------------------------------------------
// Matrix Size = 2048
// A^T.A on CPU: 6.47388e+08 GFLOPS; Time = 26537.2 msec
// A^T.A on GPU: 3.67484e+10 GFLOPS; Time = 467.5 msec
// With Unrolling 4x
// Matrix Size = 2048
// A^T.A on CPU: 6.43521e+08 GFLOPS; Time = 26696.7 msec
// A^T.A on GPU: 5.32949e+10 GFLOPS; Time = 322.355 msec
// With Unrolling 8x
// Matrix Size = 2048
// A^T.A on CPU: 6.36082e+08 GFLOPS; Time = 27008.9 msec
// A^T.A on GPU: 4.84004e+10 GFLOPS; Time = 354.953 msec
// -----------------------------------------------------
// -----------------------------------------------------
// Matrix Size = 4096
// A^T.A on CPU: 6.19884e+08 GFLOPS; Time = 221717 msec
// A^T.A on GPU: 3.27233e+10 GFLOPS; Time = 4200.03 msec
// With Unrolling 4x
// A^T.A on GPU: 4.34563e+10 GFLOPS; Time = 3162.69 msec
// With Unrolling 8x
// A^T.A on GPU: 4.58428e+10 GFLOPS; Time = 2998.05 msec
// -----------------------------------------------------
__global__ void ATAkernel(double* A, double* B) {
  int blocki = blockIdx.y * blockDim.y;
  int blockj = blockIdx.x * blockDim.x;

  int i = threadIdx.y;
  int j = threadIdx.x;

  double val = 0;

  // Allocate shared memory for block of threads
  __shared__ double Asub1[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ double Asub2[BLOCK_SIZE][BLOCK_SIZE];

  for(int k = 0; k < SIZE / BLOCK_SIZE; k++) {

    // Each thread in block bringing one element from banks
    Asub1[i][j] = A[(k * BLOCK_SIZE + i)*SIZE + (blocki + j)];
    Asub2[i][j] = A[(k * BLOCK_SIZE + i)*SIZE + (blockj + j)];

    // To ensure all element loaded
    __syncthreads();

    for(int k2 = 0; k2 < BLOCK_SIZE; k2+=8) {
      val = val + Asub1[k2][i] * Asub2[k2][j]
                + Asub1[k2+1][i] * Asub2[k2+1][j]
                + Asub1[k2+2][i] * Asub2[k2+2][j]
                + Asub1[k2+3][i] * Asub2[k2+3][j]
                + Asub1[k2+4][i] * Asub2[k2+4][j]
                + Asub1[k2+5][i] * Asub2[k2+5][j]
                + Asub1[k2+6][i] * Asub2[k2+6][j]
                + Asub1[k2+7][i] * Asub2[k2+7][j];
    }

    // Make sure all computations completed before loaded Asub again
    __syncthreads();
  }

  B[(blocki + i) * SIZE + (blockj + j)] = val;  
}

// threadsPerBlock(32, 32)
// Matrix Size = 2048
// A^T.A on CPU: 6.49936e+08 GFLOPS; Time = 26433.2 msec
// A^T.A on GPU: 3.56891e+10 GFLOPS; Time = 481.376 msec
// -----------------------------------------------------
// Matrix Size = 1024
// A^T.A on CPU: 6.51475e+08 GFLOPS; Time = 3296.34 msec
// A^T.A on GPU: 3.11944e+10 GFLOPS; Time = 68.842 msec
// -----------------------------------------------------

// Now varied threadsPerBlock (16, 64) -> (4, 256)
// Observed performance improvement for higher matrix sizes
// Matrix Size = 2048
// A^T.A on CPU: 6.47245e+08 GFLOPS; Time = 26543.1 msec
// A^T.A on GPU: 3.6516e+10 GFLOPS; Time = 470.476 msec
// -----------------------------------------------------
// __global__ void ATAkernel(double* A, double* B) {
//   int i = blockIdx.y * blockDim.y + threadIdx.y;
//   int j = blockIdx.x * blockDim.x + threadIdx.x;
  
//   double val = B[i*SIZE + j];
//   for(int k = 0; k < SIZE; k += 8) {
//     val = val
//          + A[k*SIZE + i] * A[k*SIZE + j]
//          + A[(k+1)*SIZE + i] * A[(k+1)*SIZE + j]
//          + A[(k+2)*SIZE + i] * A[(k+2)*SIZE + j]
//          + A[(k+3)*SIZE + i] * A[(k+3)*SIZE + j]
//          + A[(k+4)*SIZE + i] * A[(k+4)*SIZE + j]
//          + A[(k+5)*SIZE + i] * A[(k+5)*SIZE + j]
//          + A[(k+6)*SIZE + i] * A[(k+6)*SIZE + j]
//          + A[(k+7)*SIZE + i] * A[(k+7)*SIZE + j];
//   }
//   B[i*SIZE + j] = val;
// }

// Matrix Size = 2048
// A^T.A on CPU: 6.50094e+08 GFLOPS; Time = 26426.8 msec
// A^T.A on GPU: 3.50539e+10 GFLOPS; Time = 490.099 msec
// -----------------------------------------------------
// Matrix Size = 1024
// A^T.A on CPU: 6.52139e+08 GFLOPS; Time = 3292.98 msec
// A^T.A on GPU: 3.07095e+10 GFLOPS; Time = 69.929 msec
// -----------------------------------------------------
// __global__ void ATAkernel(double* A, double* B) {
//   int i = blockIdx.y * blockDim.y + threadIdx.y;
//   int j = blockIdx.x * blockDim.x + threadIdx.x;
  
//   double val = B[i*SIZE + j];
//   for(int k = 0; k < SIZE; ++k) {
//     val += A[k*SIZE + i] * A[k*SIZE + j];
//   }
//   B[i*SIZE + j] = val;
// }

// Matrix Size = 2048
// A^T.A on CPU: 6.49926e+08 GFLOPS; Time = 26433.6 msec
// A^T.A on GPU: 2.77876e+10 GFLOPS; Time = 618.257 msec
// -----------------------------------------------------
// Matrix Size = 1024
// A^T.A on CPU: 6.46364e+08 GFLOPS; Time = 3322.41 msec
// A^T.A on GPU: 2.25505e+10 GFLOPS; Time = 95.2298 msec
// -----------------------------------------------------
// __global__ void ATAkernel(double* A, double* B) {
//   int i = blockIdx.y * blockDim.y + threadIdx.y;
//   int j = blockIdx.x * blockDim.x + threadIdx.x;

//   for(int k = 0; k < SIZE; ++k) {
//     B[i*SIZE + j] += A[k*SIZE + i] * A[k*SIZE + j];
//   }
// }

void safe(cudaError_t status) {
  if (status != cudaSuccess) {
    cerr << cudaGetErrorString(status) << endl;
  }
}

int main() {
  cout << "Matrix Size = " << SIZE << "\n";

  double* h_in = new double[SIZE * SIZE];
  double* h_cpu_out = new double[SIZE * SIZE];
  double* h_dev_out = new double[SIZE * SIZE];

  for (int i = 0; i < SIZE; i++) {
    for (int j = 0; j < SIZE; j++) {
      h_in[i*SIZE + j] = i * j * 0.25;
      h_cpu_out[i*SIZE + j] = 0;
      h_dev_out[i*SIZE + j] = 0;
    }
  }

  double clkbegin = rtclock();
  ATAonCPU(h_in, h_cpu_out);
  double clkend = rtclock();
  double cpu_time = clkend - clkbegin;
  cout << "A^T.A on CPU: " << ((2.0 * SIZE * SIZE * SIZE) / cpu_time)
       << " GFLOPS; Time = " << cpu_time * 1000 << " msec" << endl;

  // cudaError_t status;
  cudaEvent_t start, end;
  double* d_in;
  double* d_out;
  float kernel_time;
  
  safe(cudaEventCreate(&start));
  safe(cudaEventCreate(&end));

  safe(cudaMalloc(&d_in, SIZE*SIZE*sizeof(double)));
  safe(cudaMalloc(&d_out, SIZE*SIZE*sizeof(double)));

  safe(cudaEventRecord(start, 0));

  // --- START --- //
  safe(cudaMemcpy(d_in, h_in, SIZE * SIZE * sizeof(double), cudaMemcpyHostToDevice));
  // safe(cudaMemcpy(d_out, h_dev_out, SIZE * SIZE * sizeof(double), cudaMemcpyHostToDevice));

  // dim3 threadsPerBlock(4, 256);
  // dim3 numBlocks(SIZE / threadsPerBlock.x, SIZE / threadsPerBlock.y);
  // ATAkernel<<<numBlocks, threadsPerBlock>>>(d_in, d_out);
  dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 numBlocks(SIZE / BLOCK_SIZE, SIZE / BLOCK_SIZE);
  ATAkernel<<<numBlocks, threadsPerBlock>>>(d_in, d_out);

  safe(cudaMemcpy(h_dev_out, d_out, SIZE * SIZE * sizeof(double), cudaMemcpyDeviceToHost));
  // --- END --- //

  safe(cudaEventRecord(end, 0));
  safe(cudaEventSynchronize(end));
  safe(cudaEventElapsedTime(&kernel_time, start, end));

  safe(cudaEventDestroy(start));
  safe(cudaEventDestroy(end));
  cout << "A^T.A on GPU: " << ((2.0 * SIZE * SIZE * SIZE) / (kernel_time * 1.0e-03))
       << " GFLOPS; Time = " << kernel_time << " msec" << endl;

  check_result(h_cpu_out, h_dev_out);

  return EXIT_SUCCESS;
}
