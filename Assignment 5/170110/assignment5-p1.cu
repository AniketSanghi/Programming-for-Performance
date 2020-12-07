// Compile: nvcc -g -G -arch=sm_61 -std=c++11 assignment5-p1.cu -o assignment5-p1

#include <cmath>
#include <cstdint>
#include <cuda.h>
#include <iostream>
#include <new>
#include <sys/time.h>

#define THRESHOLD (0.000001)
#define THREAD_BLOCK 64
#define BLOCK_SIZE 16

#define SIZE1 8192
#define SIZE2 8200
#define ITER 100

using std::cerr;
using std::cout;
using std::endl;

// FINAL SOLUTION PERFORMANCE
// Serial code on CPU: 5.07292e+08 GFLOPS; Time = 26457.7 msec
// Kernel 1 on GPU: 6.0965e+09 GFLOPS; Time = 2201.55 msec
// Serial code (8200) on CPU: 5.03524e+08 GFLOPS; Time = 26655.7 msec
// Kernel 2 (8200) on GPU: 1.30202e+10 GFLOPS; Time = 1032.85 msec




// Serial code on CPU: 5.10322e+08 GFLOPS; Time = 26300.6 msec
// Kernel 1 on GPU: 6.17352e+09 GFLOPS; Time = 2174.09 msec
__global__ void kernel1(double* d_k1_in) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (j < SIZE1 - 1) {
    for(int k = 0; k < ITER; k++) {
      for(int i = 1; i < (SIZE1 - 1); i++) {
        d_k1_in[i*SIZE1 + j + 1] =
              (d_k1_in[(i - 1)*SIZE1 + j + 1] + d_k1_in[i*SIZE1 + j + 1] + d_k1_in[(i + 1)*SIZE1 +j + 1]);
      }
    }
  }
}


// THREAD_BLOCK 64
// Serial code (8200) on CPU: 4.78947e+08 GFLOPS; Time = 28023.5 msec
// Kernel 2 (8200) on GPU: 1.28319e+10 GFLOPS; Time = 1048.02 msec
__global__ void kernel2(double* d_k2_in) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (j < SIZE2 - 1) {
    for(int k = 0; k < ITER; k++) {
      double prev = d_k2_in[j+1];
      for(int i = 1; i < (SIZE2 - 1); i+=4) {
        double curr = d_k2_in[i*SIZE2 + j + 1];
        double next = d_k2_in[(i+1)*SIZE2 + j + 1];
        double next2 = d_k2_in[(i+2)*SIZE2 + j + 1];

        curr = prev + curr + next;
        next = curr + next + next2;
        prev = next;

        d_k2_in[i*SIZE2 + j + 1] = curr;
        d_k2_in[(i+1)*SIZE2 + j + 1] = next;

        if(i+2 < SIZE2 - 1) {
          curr = d_k2_in[(i+3)*SIZE2 + j + 1];

          next2 = next + next2 + curr;
          curr = next2 + curr + d_k2_in[(i+4)*SIZE2 + j + 1];
          prev = curr;

          d_k2_in[(i+2)*SIZE2 + j + 1] = next2;
          d_k2_in[(i+3)*SIZE2 + j + 1] = curr;
        }
      }
    }
  }
}

// THREAD_BLOCK 128
// Serial code (8200) on CPU: 5.11297e+08 GFLOPS; Time = 26250.5 msec
// Kernel 2 (8200) on GPU: 1.16904e+10 GFLOPS; Time = 1150.35 msec
// THREAD_BLOCK 64
// Serial code (8200) on CPU: 5.13094e+08 GFLOPS; Time = 26158.5 msec
// Kernel 2 (8200) on GPU: 1.17131e+10 GFLOPS; Time = 1148.12 msec -- BEST CASE
// THREAD_BLOCK 32
// Serial code (8200) on CPU: 5.13205e+08 GFLOPS; Time = 26152.9 msec
// Kernel 2 (8200) on GPU: 1.15313e+10 GFLOPS; Time = 1166.22 msec
// __global__ void kernel2(double* d_k2_in) {
//   int j = blockIdx.x * blockDim.x + threadIdx.x;

//   if (j < SIZE2 - 1) {
//     for(int k = 0; k < ITER; k++) {
//       double prev = d_k2_in[j+1];
//       for(int i = 1; i < (SIZE2 - 1); i+=2) {
//         double curr = d_k2_in[i*SIZE2 + j + 1];
//         double next = d_k2_in[(i+1)*SIZE2 + j + 1];

//         curr = prev + curr + next;
//         next = curr + next + d_k2_in[(i+2)*SIZE2 + j + 1];
//         prev = next;

//         d_k2_in[i*SIZE2 + j + 1] = curr;
//         d_k2_in[(i+1)*SIZE2 + j + 1] = next;
//       }
//     }
//   }
// }


// Serial code on CPU: 5.07342e+08 GFLOPS; Time = 26455.1 msec
// Kernel 1 on GPU: 8.5619e+09 GFLOPS; Time = 1567.62 msec
// Serial code (8200) on CPU: 5.14827e+08 GFLOPS; Time = 26070.5 msec
// Kernel 2 (8200) on GPU: 5.2163e+09 GFLOPS; Time = 2578.07 msec
// __global__ void kernel2(double* d_k2_in) {
//   int j = blockIdx.x * blockDim.x + threadIdx.x;

//   if (j < SIZE2 - 1) {
//     for(int kit = 0; kit < ITER; kit++) {

//       for(int i = 1; i < SIZE1+1; i+=BLOCK_SIZE) {

//         double A[BLOCK_SIZE+2];

//         for(int k = 0; k < BLOCK_SIZE+2; ++k) {
//           A[k] = d_k2_in[(i+k-1)*SIZE2 + j + 1];
//         }

//         for(int k = 1; k < BLOCK_SIZE+1; ++k) {
//           A[k] = A[k-1] + A[k] + A[k+1];
//         }

//         for(int k = 0; k < BLOCK_SIZE; ++k) {
//           d_k2_in[(i+k)*SIZE2 + j + 1] = A[k+1];
//         }
//       }

//       #pragma unroll
//       for(int i = SIZE1 + 1; i < (SIZE2 - 1); i++) {
//         d_k2_in[i*SIZE2 + j + 1] =
//               (d_k2_in[(i - 1)*SIZE2 + j + 1] + d_k2_in[i*SIZE2 + j + 1] + d_k2_in[(i + 1)*SIZE2 + j + 1]);
//       }

//     }
//   }
// }

// THREAD_BLOCK 1024
// Serial code (8200) on CPU: 5.14113e+08 GFLOPS; Time = 26106.7 msec
// Kernel 2 (8200) on GPU: 6.15432e+09 GFLOPS; Time = 2185.13 msec
// THREAD_BLOCK 512
// Serial code (8200) on CPU: 5.13975e+08 GFLOPS; Time = 26113.7 msec
// Kernel 2 (8200) on GPU: 8.3103e+09 GFLOPS; Time = 1618.23 msec
// THEAD_BLOCK 256
// Serial code (8200) on CPU: 5.1201e+08 GFLOPS; Time = 26213.9 msec
// Kernel 2 (8200) on GPU: 8.5175e+09 GFLOPS; Time = 1578.87 msec
// THEAD_BLOCK 128
// Serial code (8200) on CPU: 5.11713e+08 GFLOPS; Time = 26229.1 msec
// Kernel 2 (8200) on GPU: 8.76146e+09 GFLOPS; Time = 1534.9 msec
// THREAD_BLOCK 64
// Serial code (8200) on CPU: 5.14438e+08 GFLOPS; Time = 26090.2 msec
// Kernel 2 (8200) on GPU: 8.70571e+09 GFLOPS; Time = 1544.73 msec   --- BEST CASE
// THREAD_BLOCK 32
// Serial code (8200) on CPU: 5.14403e+08 GFLOPS; Time = 26091.9 msec
// Kernel 2 (8200) on GPU: 8.11434e+09 GFLOPS; Time = 1657.31 msec
// __global__ void kernel2(double* d_k2_in) {
//   int j = blockIdx.x * blockDim.x + threadIdx.x;

//   if (j < SIZE2 - 1) {
//     for(int k = 0; k < ITER; k++) {
//       for(int i = 1; i < (SIZE2 - 1); i++) {
//         d_k2_in[i*SIZE2 + j + 1] =
//               (d_k2_in[(i - 1)*SIZE2 + j + 1] + d_k2_in[i*SIZE2 + j + 1] + d_k2_in[(i + 1)*SIZE2 +j + 1]);
//       }
//     }
//   }
// }

__host__ void serial(double** h_ser_in) {
  for (int k = 0; k < ITER; k++) {
    for (int i = 1; i < (SIZE1 - 1); i++) {
      for (int j = 0; j < (SIZE1 - 1); j++) {
        h_ser_in[i][j + 1] =
            (h_ser_in[i - 1][j + 1] + h_ser_in[i][j + 1] + h_ser_in[i + 1][j + 1]);
      }
    }
  }
}

__host__ void serial2(double** h_ser_in) {
  for (int k = 0; k < ITER; k++) {
    for (int i = 1; i < (SIZE2 - 1); i++) {
      for (int j = 0; j < (SIZE2 - 1); j++) {
        h_ser_in[i][j + 1] =
            (h_ser_in[i - 1][j + 1] + h_ser_in[i][j + 1] + h_ser_in[i + 1][j + 1]);
      }
    }
  }
}

__host__ void check_result(double** w_ref, double* w_opt, uint64_t size) {
  double maxdiff = 0.0, this_diff = 0.0;
  int numdiffs = 0;

  for (uint64_t i = 0; i < size; i++) {
    for (uint64_t j = 0; j < size; j++) {
      this_diff = w_ref[i][j] - w_opt[i*size + j];
      if (fabs(this_diff) > THRESHOLD) {
        numdiffs++;
        if (this_diff > maxdiff)
          maxdiff = this_diff;
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
  double** h_ser_in = new double*[SIZE1];

  for (int i = 0; i < SIZE1; i++) {
    h_ser_in[i] = new double[SIZE1];
  }

  double* h_k1_in = new double[SIZE1*SIZE1];

  for (int i = 0; i < SIZE1; i++) {
    for (int j = 0; j < SIZE1; j++) {
      h_ser_in[i][j] = 1;
      h_k1_in[i*SIZE1 + j] = 1;
    }
  }

  double** h_ser2_in = new double*[SIZE2];

  for (int i = 0; i < SIZE2; i++) {
    h_ser2_in[i] = new double[SIZE2];
  }

  double* h_k2_in = new double[SIZE2*SIZE2];

  for (int i = 0; i < SIZE2; i++) {
    for (int j = 0; j < SIZE2; j++) {
      h_ser2_in[i][j] = 1;
      h_k2_in[i*SIZE2 + j] = 1;
    }
  }

  double clkbegin = rtclock();
  serial(h_ser_in);
  double clkend = rtclock();
  double time = clkend - clkbegin; // seconds
  cout << "Serial code on CPU: " << ((2.0 * SIZE1 * SIZE1 * ITER) / time)
       << " GFLOPS; Time = " << time * 1000 << " msec" << endl;

  cudaError_t status;
  cudaEvent_t start, stop;
  float k1_time; // ms

  double* d_k1_in;

  safe(status = cudaEventCreate(&start));
  safe(status = cudaEventCreate(&stop));

  safe(status = cudaMalloc(&d_k1_in, SIZE1*SIZE1*sizeof(double)));

  safe(status = cudaEventRecord(start,0));

  // --- START --- //
  safe(status = cudaMemcpy(d_k1_in, h_k1_in, SIZE1*SIZE1*sizeof(double), cudaMemcpyHostToDevice));

  dim3 theadsPerblock(1024);
  dim3 numBlocks(SIZE1 / theadsPerblock.x);
  kernel1<<<numBlocks, theadsPerblock>>>(d_k1_in);

  safe(status = cudaMemcpy(h_k1_in, d_k1_in, SIZE1*SIZE1*sizeof(double), cudaMemcpyDeviceToHost));
  // --- END --- //

  safe(status = cudaEventRecord(stop,0));
  safe(status = cudaEventSynchronize(stop));
  safe(status = cudaEventElapsedTime(&k1_time,start,stop));
 
  check_result(h_ser_in, h_k1_in, SIZE1);
  cout << "Kernel 1 on GPU: " << ((2.0 * SIZE1 * SIZE1 * ITER) / (k1_time * 1.0e-3))
       << " GFLOPS; Time = " << k1_time << " msec" << endl;

  safe(status = cudaEventDestroy(start));
  safe(status = cudaEventDestroy(stop));


  clkbegin = rtclock();
  serial2(h_ser2_in);
  clkend = rtclock();
  time = clkend - clkbegin; // seconds
  cout << "Serial code (8200) on CPU: " << ((2.0 * SIZE1 * SIZE1 * ITER) / time)
       << " GFLOPS; Time = " << time * 1000 << " msec" << endl;



  double* d_k2_in;

  safe(status = cudaEventCreate(&start));
  safe(status = cudaEventCreate(&stop));

  safe(status = cudaMalloc(&d_k2_in, SIZE2*SIZE2*sizeof(double)));

  safe(status = cudaEventRecord(start,0));

  // --- START --- //
  safe(status = cudaMemcpy(d_k2_in, h_k2_in, SIZE2*SIZE2*sizeof(double), cudaMemcpyHostToDevice));

  dim3 theadsPerblock2(THREAD_BLOCK);
  dim3 numBlocks2((SIZE2 + theadsPerblock2.x - 1) / theadsPerblock2.x);
  kernel2<<<numBlocks2, theadsPerblock2>>>(d_k2_in);

  safe(status = cudaMemcpy(h_k2_in, d_k2_in, SIZE2*SIZE2*sizeof(double), cudaMemcpyDeviceToHost));
  // --- END --- //

  safe(status = cudaEventRecord(stop,0));
  safe(status = cudaEventSynchronize(stop));
  safe(status = cudaEventElapsedTime(&k1_time,start,stop));

  check_result(h_ser2_in, h_k2_in, SIZE2);
  cout << "Kernel 2 (8200) on GPU: " << ((2.0 * SIZE2 * SIZE2 * ITER) / (k1_time * 1.0e-3))
       << " GFLOPS; Time = " << k1_time << " msec" << endl;

  safe(status = cudaEventDestroy(start));
  safe(status = cudaEventDestroy(stop));

  safe(cudaFree(d_k1_in));
  safe(cudaFree(d_k2_in));

  if (status != cudaSuccess) {
    cerr << cudaGetErrorString(status) << endl;
  }

  delete[] h_ser_in;
  delete[] h_k1_in;

  delete[] h_k2_in;

  return EXIT_SUCCESS;
}
