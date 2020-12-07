// Compile: g++ -O2 -o problem2 problem2.cpp
// Execute: ./problem2

#include <cmath>
#include <iostream>
#include <sys/time.h>
#include <unistd.h>
#include <xmmintrin.h>

using namespace std;

const int N = 1024;
const int Niter = 10;
const double THRESHOLD = 0.0000001;

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

void reference(double** A, double** B, double** C) {
  int i, j, k;
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      for (k = 0; k < i + 1; k++) {
        C[i][j] += A[k][i] * B[j][k];
      }
    }
  }
}

void check_result(double** w_ref, double** w_opt) {
  double maxdiff, this_diff;
  int numdiffs;
  int i, j;
  numdiffs = 0;
  maxdiff = 0;

  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      this_diff = w_ref[i][j] - w_opt[i][j];
      if (fabs(this_diff) > THRESHOLD) {
        numdiffs++;
        if (this_diff > maxdiff)
          maxdiff = this_diff;
      }
    }
  }

  if (numdiffs > 0) {
    cout << numdiffs << " Diffs found over threshold " << THRESHOLD << "; Max Diff = " << maxdiff
         << endl;
  } else {
    cout << "No differences found between base and test versions\n";
  }
}

// TODO: THIS IS INITIALLY IDENTICAL TO REFERENCE. MAKE YOUR CHANGES TO OPTIMIZE THIS FUNCTION
// You can create multiple versions of the optimized() function to test your changes
// i,j UNROLLING AND JAMMING + PEELING
void optimized(double** A, double** B, double** C) {
  int i, j, k;
  for (i = 0; i < N; i+=2) {
    for (j = 0; j < N; j+=4) {
      double t1 = C[i][j], t2 = C[i][j+1], t3 = C[i][j+2], t4 = C[i][j+3];
      double t11 = C[i+1][j], t21 = C[i+1][j+1], t31 = C[i+1][j+2], t41 = C[i+1][j+3];
      for (k = 0; k < i + 1; k++) {
        t1 += A[k][i] * B[j][k];
        t2 += A[k][i] * B[j+1][k];
        t3 += A[k][i] * B[j+2][k];
        t4 += A[k][i] * B[j+3][k];
        t11 += A[k][i+1] * B[j][k];
        t21 += A[k][i+1] * B[j+1][k];
        t31 += A[k][i+1] * B[j+2][k];
        t41 += A[k][i+1] * B[j+3][k];
      }
      t11 += A[i+1][i+1] * B[j][i+1];
      t21 += A[i+1][i+1] * B[j+1][i+1];
      t31 += A[i+1][i+1] * B[j+2][i+1];
      t41 += A[i+1][i+1] * B[j+3][i+1];

      C[i][j] = t1;
      C[i][j+1] = t2;
      C[i][j+2] = t3;
      C[i][j+3] = t4;
      C[i+1][j] = t11;
      C[i+1][j+1] = t21;
      C[i+1][j+2] = t31;
      C[i+1][j+3] = t41;
    }
  }
}

void optimized_intrinsics(double** A, double** B, double** C) {
  int i, j, k;
  for (i = 0; i < N; i+=2) {
    for (j = 0; j < N; j+=4) {
      __m128d C00, C01, C10, C11;
      C00 = _mm_load_pd(&C[i][j]);
      C01 = _mm_load_pd(&C[i][j+2]);
      C10 = _mm_load_pd(&C[i+1][j]);
      C11 = _mm_load_pd(&C[i+1][j+2]);

      for (k = 0; k < i + 1; k++) {
        __m128d B00, B01, B02, B03;
        __m128d A00, A01;
        B01 = _mm_set_pd1(B[j+1][k]);
        B00 = _mm_loadl_pd(B01,&B[j][k]);
        B03 = _mm_set_pd1(B[j+3][k]);
        B02 = _mm_loadl_pd(B03,&B[j+2][k]);
        A00 = _mm_set_pd1(A[k][i]);
        A01 = _mm_set_pd1(A[k][i+1]);

        __m128d P0, P1, P2, P3;
        P0 = _mm_mul_pd(A00, B00);
        P1 = _mm_mul_pd(A00, B02);
        P2 = _mm_mul_pd(A01, B00);
        P3 = _mm_mul_pd(A01, B02);

        C00 = _mm_add_pd(C00, P0);
        C01 = _mm_add_pd(C01, P1);
        C10 = _mm_add_pd(C10, P2);
        C11 = _mm_add_pd(C11, P3);
      }

      __m128d A00;
      __m128d B00, B01, B02, B03;
      A00 = _mm_set_pd1(A[i+1][i+1]);
      B01 = _mm_set_pd1(B[j+1][i+1]);
      B00 = _mm_loadl_pd(B01,&B[j][i+1]);
      B03 = _mm_set_pd1(B[j+3][i+1]);
      B02 = _mm_loadl_pd(B03,&B[j+2][i+1]);

      __m128d P0, P1;
      P0 = _mm_mul_pd(A00, B00);
      P1 = _mm_mul_pd(A00, B02);

      C10 = _mm_add_pd(C10, P0);
      C11 = _mm_add_pd(C11, P1);

      _mm_store_pd(&C[i][j], C00);
      _mm_store_pd(&C[i][j+2], C01);
      _mm_store_pd(&C[i+1][j], C10);
      _mm_store_pd(&C[i+1][j+2], C11);
    }
  }
}

// 4 - Loop unrolling + Jamming
// GCC - 2.4X
// ICC - 2.5X
// void optimized(double** A, double** B, double** C) {
//   int i, j, k;
//   for (i = 0; i < N; i++) {
//     for (j = 0; j < N; j+=8) {
//       double t1 = C[i][j], t2 = C[i][j+1], t3 = C[i][j+2], t4 = C[i][j+3];
//       double t11 = C[i][j+4], t21 = C[i][j+5], t31 = C[i][j+6], t41 = C[i][j+7];
//       for (k = 0; k < i + 1; k++) {
//         t1 += A[k][i] * B[j][k];
//         t2 += A[k][i] * B[j+1][k];
//         t3 += A[k][i] * B[j+2][k];
//         t4 += A[k][i] * B[j+3][k];
//         t11 += A[k][i] * B[j+4][k];
//         t21 += A[k][i] * B[j+5][k];
//         t31 += A[k][i] * B[j+6][k];
//         t41 += A[k][i] * B[j+7][k];
//       }
//       C[i][j] = t1;
//       C[i][j+1] = t2;
//       C[i][j+2] = t3;
//       C[i][j+3] = t4;
//       C[i][j+4] = t11;
//       C[i][j+5] = t21;
//       C[i][j+6] = t31;
//       C[i][j+7] = t41;
//     }
//   }
// }

// 3 - k i j permutation (GCC degrades, ICC upgrades)
// GCC - 0.94X
// ICC - 1.2X
// void optimized(double** A, double** B, double** C) {
//   int i, j, k;
//   for (k = 0; k < N; k++) {
//     for (i = k; i < N; i++) {
//       for (j = 0; j < N; j++) {
//         C[i][j] += A[k][i] * B[j][k];
//       }
//     }
//   }
// }

// 2 - Taking Benefit of reduction (ICC shows improvement)
// GCC - 1X
// ICC - 1.2X
// void optimized(double** A, double** B, double** C) {
//   int i, j, k;
//   for (i = 0; i < N; i++) {
//     for (j = 0; j < N; j++) {
//       double temp = C[i][j];
//       for (k = 0; k < i + 1; k++) {
//         temp += A[k][i] * B[j][k];
//       }
//       C[i][j] = temp;
//     }
//   }
// }

// 1 - Blocking i,j (ICC shows better improvement)
// GCC - 1.04X
// ICC - 1.3X
// void optimized(double** A, double** B, double** C) {
//   int i, j, k, i2, j2;
//   int BS = 16;
//   for (i = 0; i < N; i+=BS) {
//     for (j = 0; j < N; j+=BS) {
//       for(i2 = i; i2 < i + BS; ++i2) {
//         for(j2 = j; j2 < j + BS; ++j2) {
//           for (k = 0; k < i2 + 1; k++) {
//             C[i2][j2] += A[k][i2] * B[j2][k];
//           }
//         }
//       }
//     }
//   }
// }


int main() {
  double clkbegin, clkend;
  double t;

  int i, j, it;
  cout.setf(ios::fixed, ios::floatfield);
  cout.precision(5);

  double **A, **B, **C_ref, **C_opt;
  A = new double*[N];
  B = new double*[N];
  C_ref = new double*[N];
  C_opt = new double*[N];
  for (i = 0; i < N; i++) {
    A[i] = new double[N];
    B[i] = new double[N];
    C_ref[i] = new double[N];
    C_opt[i] = new double[N];
  }

  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      A[i][j] = i + j + 1;
      B[i][j] = (i + 1) * (j + 1);
      C_ref[i][j] = 0.0;
      C_opt[i][j] = 0.0;
    }
  }

  clkbegin = rtclock();
  for (it = 0; it < Niter; it++)
    reference(A, B, C_ref);
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Reference Version: Matrix Size = " << N << ", " << 2.0 * 1e-9 * N * N * Niter / t
       << " GFLOPS; Time = " << t / Niter << " sec\n\n";

  clkbegin = rtclock();
  for (it = 0; it < Niter; it++)
    optimized(A, B, C_opt);
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Optimized Version: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
  check_result(C_ref, C_opt);
  cout<<"\n";

  // Reset
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      C_opt[i][j] = 0.0;
    }
  }

  clkbegin = rtclock();
  for (it = 0; it < Niter; it++)
    optimized_intrinsics(A, B, C_opt);
  clkend = rtclock();
  t = clkend - clkbegin;
  cout << "Optimized Version with Intrinsics: Matrix Size = " << N << ", Time = " << t / Niter << " sec\n";
  check_result(C_ref, C_opt);

  // Reset
  for (i = 0; i < N; i++) {
    for (j = 0; j < N; j++) {
      C_opt[i][j] = 0.0;
    }
  }

  return EXIT_SUCCESS;
}
