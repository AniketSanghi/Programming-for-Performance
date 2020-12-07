// Compile: g++ -O2 -fopenmp -o problem3 problem3.cpp
// Execute: ./problem3

#include <cassert>
#include <iostream>
#include <omp.h>

#define N (1 << 12)
#define ITER 100

using namespace std;

void check_result(uint32_t** w_ref, uint32_t** w_opt) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      assert(w_ref[i][j] == w_opt[i][j]);
    }
  }
  cout << "No differences found between base and test versions\n";
}

void reference(uint32_t** A) {
  int i, j, k;
  for (k = 0; k < ITER; k++) {
    for (i = 1; i < N; i++) {
      for (j = 0; j < (N - 1); j++) {
        A[i][j + 1] = A[i - 1][j + 1] + A[i][j + 1];
      }
    }
  }
}

// TODO: MAKE YOUR CHANGES TO OPTIMIZE THIS FUNCTION
// 6.5X SpeedUp
void omp_version(uint32_t** A) {
  int i, j, k, j2, i2;
  int BS = 16;
  #pragma omp parallel for
  for (j = 0; j < (N - 1); j+=BS) {
    for (k = 0; k < ITER; k++) {
      for (i = 1; i < N; i+=BS) {
        for(j2 = j; j2 < min(j+BS, N-1); j2+=4) {
          for(i2 = i; i2 < min(i+BS, N); i2++) {
            A[i2][j2 + 1] = A[i2 - 1][j2 + 1] + A[i2][j2 + 1];
            A[i2][j2 + 2] = A[i2 - 1][j2 + 2] + A[i2][j2 + 2];
            A[i2][j2 + 3] = A[i2 - 1][j2 + 3] + A[i2][j2 + 3];
            A[i2][j2 + 4] = A[i2 - 1][j2 + 4] + A[i2][j2 + 4];
          }
        }
      }
    }
  }
}
// BlockSize SpeedUp
// 1024 1.3 
// 512 0.76
// 256 0.45
// 128 0.43
// 64 0.29
// 32 0.26
// 16 0.24 // Un-2 0.19 // Un-4 0.16 // Un-8 0.199
// 8 0.25

// k, j, i permutation with j blocking (Around 2X speedup)
// void omp_version(uint32_t** A) {
//   int i, j, k, i2, j2;
//   int BS = 512;
//   for (k = 0; k < ITER; k++) {
//     #pragma omp parallel for
//     for (j = 0; j < (N - 1); j+=BS) {
//       for (i = 1; i < N; i++) {
//         for(j2 = j; j2 < min(j+BS, N-1); ++j2) {
//           A[i][j2 + 1] = A[i - 1][j2 + 1] + A[i][j2 + 1];
//         }
//       }
//     }
//   }
// }

// j, i, k permutation with j blocking (2X speedup)
// void omp_version(uint32_t** A) {
//   int i, j, k, j2;
//   int BS = 1024;
//   #pragma omp parallel for schedule(dynamic, 1) num_threads(4) private(j, k, i)
//   for (j = 0; j < (N - 1); j+=BS) {
//     for (k = 0; k < ITER; k++) {
//       for (i = 1; i < N; i++) {
//         for(j2 = j; j2 < min(j+BS, N-1); j2++) {
//           A[i][j2 + 1] = A[i - 1][j2 + 1] + A[i][j2 + 1];
//         }
//       }
//     }
//   }
// }

// j, i, k permutation with Parallelism (Not enough improvement)
// void omp_version(uint32_t** A) {
//   int i, j, k;
//   #pragma omp parallel for schedule(dynamic, 64) num_threads(12) private(k, i, j) shared(A)
//   for (j = 0; j < (N-1); j++) {
//     for (k = 0; k < ITER; k++) {
//       for (i = 1; i < N; i++) {
//         A[i][j + 1] = A[i - 1][j + 1] + A[i][j + 1];
//       }
//     }
//   }
// }


int main() {
  uint32_t** A_ref = new uint32_t*[N];
  for (int i = 0; i < N; i++) {
    A_ref[i] = new uint32_t[N];
  }

  uint32_t** A_omp = new uint32_t*[N];
  for (int i = 0; i < N; i++) {
    A_omp[i] = new uint32_t[N];
  }

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A_ref[i][j] = i + j + 1;
      A_omp[i][j] = i + j + 1;
    }
  }

  double start = omp_get_wtime();
  reference(A_ref);
  double end = omp_get_wtime();
  cout << "Time for reference version: " << end - start << " seconds\n";

  start = omp_get_wtime();
  omp_version(A_omp);
  end = omp_get_wtime();
  cout << "Version1: Time with OpenMP: " << end - start << " seconds\n";
  check_result(A_ref, A_omp);

  // Reset
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A_omp[i][j] = i + j + 1;
    }
  }

  // Another optimized version possibly

  return EXIT_SUCCESS;
}
