// Compile: g++ -O2 -o problem1 problem1.cpp
// Execute: ./problem1

#include <cmath>
#include <iostream>
#include <sys/time.h>
#include <unistd.h>
#include <xmmintrin.h>
#include <immintrin.h>

using namespace std;

const int N = 1 << 13;
const int Niter = 10;
const double THRESHOLD = 0.000001;

double rtclock() {
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0) {
    cout << "Error return from gettimeofday: " << stat << endl;
  }
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void reference(double** A, double* x, double* y_ref, double* z_ref) {
  int i, j;
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      y_ref[j] = y_ref[j] + A[i][j] * x[i];
      z_ref[j] = z_ref[j] + A[j][i] * x[i];
    }
  }
}

void check_result(double* w_ref, double* w_opt) {
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

// TODO: INITIALLY IDENTICAL TO REFERENCE; MAKE YOUR CHANGES TO OPTIMIZE THIS CODE
// You can create multiple versions of the optimized() function to test your changes
void optimized(double** A, double* x, double* y_opt, double* z_opt) {
  int i, j;
  for (i = 0; i < N; i+=2) {
    for (j = 0; j < N; j++) {
      y_opt[j] = y_opt[j] + A[i][j] * x[i]
                          + A[i+1][j] * x[i+1];
    }
  }
  for (j = 0; j < N; j++) {
    for (i = 0; i < N; i+=8) {
      z_opt[j] = z_opt[j] + A[j][i] * x[i]
                          + A[j][i+1] * x[i+1]
                          + A[j][i+2] * x[i+2]
                          + A[j][i+3] * x[i+3]
                          + A[j][i+4] * x[i+4]
                          + A[j][i+5] * x[i+5]
                          + A[j][i+6] * x[i+6]
                          + A[j][i+7] * x[i+7];
    }
  }
}


// 6 - Loop Split + Interchange + Unrolling + Tiling (ICC BEST)
void optimized_icc_best(double** A, double* x, double* y_opt, double* z_opt) {
  int i, j, i2;
  int BLOCK_SIZE = 16;
  for (i = 0; i < N; i+=BLOCK_SIZE) {
    for (j = 0; j < N; ++j) {
      for(i2 = i; i2 < i+BLOCK_SIZE; ++i2) {
        y_opt[j] = y_opt[j] + A[i2][j] * x[i2];
      }
    }
  }
  for (j = 0; j < N; j++) {
    for (i = 0; i < N; i+=8) {
      z_opt[j] = z_opt[j] + A[j][i] * x[i]
                          + A[j][i+1] * x[i+1]
                          + A[j][i+2] * x[i+2]
                          + A[j][i+3] * x[i+3]
                          + A[j][i+4] * x[i+4]
                          + A[j][i+5] * x[i+5]
                          + A[j][i+6] * x[i+6]
                          + A[j][i+7] * x[i+7];
    }
  }
}


void optimized_intrinsics(double** A, double* x, double* y_opt, double* z_opt) {
  int i, j, i2;
  for (i = 0; i < N; i+=2) {
    for (j = 0; j < N; j+=2) {
      __m128d rY, rA, rX, rP, rQ;
      rY = _mm_load_pd(&y_opt[j]);
      rA = _mm_load_pd(&A[i][j]);
      rX = _mm_set_pd1(x[i]);
      rP = _mm_mul_pd(rA, rX);
      rQ = _mm_add_pd(rY, rP);

      rA = _mm_load_pd(&A[i+1][j]);
      rX = _mm_set_pd1(x[i+1]);
      rP = _mm_mul_pd(rA, rX);
      rQ = _mm_add_pd(rQ, rP);
      _mm_store_pd(&y_opt[j], rQ);
    }
  }
  for (j = 0; j < N; j++) {
    for (i = 0; i < N; i+=8) {
      __m128d rA, rX, rP, rQ, rR, rC, rZ;
      rA = _mm_load_pd(&A[j][i]);
      rX = _mm_load_pd(&x[i]);
      rQ = _mm_mul_pd(rA, rX);
      rA = _mm_load_pd(&A[j][i+2]);
      rX = _mm_load_pd(&x[i+2]);
      rP = _mm_mul_pd(rA, rX);
      rQ = _mm_add_pd(rP, rQ);
      rA = _mm_load_pd(&A[j][i+4]);
      rX = _mm_load_pd(&x[i+4]);
      rP = _mm_mul_pd(rA, rX);
      rQ = _mm_add_pd(rP, rQ);
      rA = _mm_load_pd(&A[j][i+6]);
      rX = _mm_load_pd(&x[i+6]);
      rP = _mm_mul_pd(rA, rX);
      rQ = _mm_add_pd(rP, rQ);
      rZ = _mm_set_pd1(z_opt[j]);
      rR = _mm_hadd_pd(rQ, rA);
      rC = _mm_add_pd(rR, rZ);
      _mm_storeh_pd(&z_opt[j], rC);
    }
  }
}

// 4 - Loop Split + Interchange for 2 (ICC BEST / GCC WORST !)
// GCC gave around 1.12X
// ICC gave 30X !!
// void optimized(double** A, double* x, double* y_opt, double* z_opt) {
//   int i, j;
//   for (j = 0; j < N; j++) {
//     for (i = 0; i < N; i++) {
//       y_opt[j] = y_opt[j] + A[i][j] * x[i];
//     }
//   }
//   for (j = 0; j < N; j++) {
//     for (i = 0; i < N; i+=8) {
//       z_opt[j] = z_opt[j] + A[j][i] * x[i]
//                           + A[j][i+1] * x[i+1]
//                           + A[j][i+2] * x[i+2]
//                           + A[j][i+3] * x[i+3]
//                           + A[j][i+4] * x[i+4]
//                           + A[j][i+5] * x[i+5]
//                           + A[j][i+6] * x[i+6]
//                           + A[j][i+7] * x[i+7];
//     }
//   }
// }

// 3 - Loop Split + Interchange for 1
// GCC - 5.3X
// ICC - 11X
// void optimized(double** A, double* x, double* y_opt, double* z_opt) {
//   int i, j;
//   for (i = 0; i < N; i++) {
//     for (j = 0; j < N; j++) {
//       y_opt[j] = y_opt[j] + A[i][j] * x[i];
//     }
//   }
//   for (j = 0; j < N; j++) {
//     for (i = 0; i < N; i++) {
//       z_opt[j] = z_opt[j] + A[j][i] * x[i];
//     }
//   }
// }

// 2 - Loop Interchange + Loop Tiling
// GCC - 1.75X
// ICC - 3X
// void optimized(double** A, double* x, double* y_opt, double* z_opt) {
//   int i, j, i2, j2;
//   int BLOCK_SIZE = 256;
//   for (j = 0; j < N; j+=BLOCK_SIZE) {
//     for (i = 0; i < N; i+=BLOCK_SIZE) {
//       for(j2 = j; j2 < j+BLOCK_SIZE; ++j2) {
//         for(i2 = i; i2 < i+BLOCK_SIZE; ++i2) {
//           y_opt[j2] = y_opt[j2] + A[i2][j2] * x[i2];
//           z_opt[j2] = z_opt[j2] + A[j2][i2] * x[i2];
//         }
//       }
//     }
//   }
// }

// 1 - Loop Interchange
// GCC - 1.1X
// ICC - 8.2X
// void optimized(double** A, double* x, double* y_opt, double* z_opt) {
//   int i, j;
//   for (j = 0; j < N; j++) {
//     for (i = 0; i < N; i++) {
//       y_opt[j] = y_opt[j] + A[i][j] * x[i];
//       z_opt[j] = z_opt[j] + A[j][i] * x[i];
//     }
//   }
// }

int main() {
  double clkbegin, clkend;
  double t;

  int i, j, it;
  cout.setf(ios::fixed, ios::floatfield);
  cout.precision(5);

  double** A;
  A = new double*[N];
  for (int i = 0; i < N; i++) {
    A[i] = new double[N];
  }

  double *x, *y_ref, *z_ref, *y_opt, *z_opt;
  x = new double[N];
  y_ref = new double[N];
  z_ref = new double[N];
  y_opt = new double[N];
  z_opt = new double[N];

  for (i = 0; i < N; i++) {
    x[i] = i;
    y_ref[i] = 1.0;
    y_opt[i] = 1.0;
    z_ref[i] = 2.0;
    z_opt[i] = 2.0;
    for (j = 0; j < N; j++) {
      A[i][j] = (i + 2.0 * j) / (2.0 * N);
    }
  }

  clkbegin = rtclock();
  for (it = 0; it < Niter; it++) {
    reference(A, x, y_ref, z_ref);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Reference Version: Matrix Size = " << N << ", " << 4.0 * 1e-9 * N * N * Niter / t
       << " GFLOPS; Time = " << t / Niter << " sec\n\n";

  clkbegin = rtclock();
  for (it = 0; it < Niter; it++) {
    optimized(A, x, y_opt, z_opt);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Optimized Version: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
  check_result(y_ref, y_opt);
  cout<<"\n";

  // Reset
  for (i = 0; i < N; i++) {
    y_opt[i] = 1.0;
    z_opt[i] = 2.0;
  }

  // Another optimized version possibly

  clkbegin = rtclock();
  for (it = 0; it < Niter; it++) {
    optimized_icc_best(A, x, y_opt, z_opt);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Optimized Version best with ICC: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
  check_result(y_ref, y_opt);
  cout<<"\n";

  // Reset
  for (i = 0; i < N; i++) {
    y_opt[i] = 1.0;
    z_opt[i] = 2.0;
  }


  clkbegin = rtclock();
  for (it = 0; it < Niter; it++) {
    optimized_intrinsics(A, x, y_opt, z_opt);
  }
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Optimized Version with Intrinsics: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
  check_result(y_ref, y_opt);
  cout<<"\n";

  // Reset
  for (i = 0; i < N; i++) {
    y_opt[i] = 1.0;
    z_opt[i] = 2.0;
  }

  return EXIT_SUCCESS;
}
