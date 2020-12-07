/**
 * g++ -o problem5 problem5.cpp -lpthread
 * ./problem5
 */

// TODO: This file is just a template, feel free to modify it to suit your needs

#include <cstring>
#include <iostream>
#include <pthread.h>
#include <sys/time.h>

using std::cout;
using std::endl;

const uint16_t MAT_SIZE = 4096;

void sequential_matmul();
void parallel_matmul();
// TODO: Other function definitions
void sequential_matmul_opt();
void parallel_matmul_opt();

double rtclock();
void check_result(uint64_t*, uint64_t*);
const double THRESHOLD = 0.0000001;

uint64_t* matrix_A;
uint64_t* matrix_B;
uint64_t* sequential_C;
uint64_t* sequential_opt_C;
uint64_t* parallel_C;
uint64_t* parallel_opt_C;

uint16_t block_size;
uint16_t NUM_THREADS;

void* parallel_matmul(void* thread_id) {
  long t = (long)thread_id;
  long section = MAT_SIZE/NUM_THREADS;

  int i, j, k;
  for (i = t*section; i < (t+1)*section; i++) {
    for (j = 0; j < MAT_SIZE; j++) {
      uint16_t temp = 0;
      for (k = 0; k < MAT_SIZE; k++)
        temp += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j]);
      parallel_C[i * MAT_SIZE + j] = temp;
    }
  }
  return 0;
}

void sequential_matmul() {
  int i, j, k;
  for (i = 0; i < MAT_SIZE; i++) {
    for (j = 0; j < MAT_SIZE; j++) {
      uint16_t temp = 0;
      for (k = 0; k < MAT_SIZE; k++)
        temp += (matrix_A[i * MAT_SIZE + k] * matrix_B[k * MAT_SIZE + j]);
      sequential_C[i * MAT_SIZE + j] = temp;
    }
  }
}

void sequential_matmul_opt() {
  for(int i = 0; i<MAT_SIZE; i+=block_size) {
    for(int j = 0; j<MAT_SIZE; j+=block_size) {
      for(int k = 0; k<MAT_SIZE; k+=block_size) {
        for(int i2 = i; i2<i+block_size; ++i2) {
          for(int j2 = j; j2<j+block_size; ++j2) {
            for(int k2 = k; k2<k+block_size; k2+=16) {
              sequential_opt_C[i2 * MAT_SIZE + j2] += (matrix_A[i2 * MAT_SIZE + k2] * matrix_B[k2 * MAT_SIZE + j2]) 
                                                    + (matrix_A[i2 * MAT_SIZE + k2 + 1] * matrix_B[(k2 + 1) * MAT_SIZE + j2])
                                                    + (matrix_A[i2 * MAT_SIZE + k2 + 2] * matrix_B[(k2 + 2) * MAT_SIZE + j2])
                                                    + (matrix_A[i2 * MAT_SIZE + k2 + 3] * matrix_B[(k2 + 3) * MAT_SIZE + j2])
                                                    + (matrix_A[i2 * MAT_SIZE + k2 + 4] * matrix_B[(k2 + 4) * MAT_SIZE + j2])
                                                    + (matrix_A[i2 * MAT_SIZE + k2 + 5] * matrix_B[(k2 + 5) * MAT_SIZE + j2])
                                                    + (matrix_A[i2 * MAT_SIZE + k2 + 6] * matrix_B[(k2 + 6) * MAT_SIZE + j2])
                                                    + (matrix_A[i2 * MAT_SIZE + k2 + 7] * matrix_B[(k2 + 7) * MAT_SIZE + j2])
                                                    + (matrix_A[i2 * MAT_SIZE + k2 + 8] * matrix_B[(k2 + 8) * MAT_SIZE + j2])
                                                    + (matrix_A[i2 * MAT_SIZE + k2 + 9] * matrix_B[(k2 + 9) * MAT_SIZE + j2])
                                                    + (matrix_A[i2 * MAT_SIZE + k2 + 10] * matrix_B[(k2 + 10) * MAT_SIZE + j2])
                                                    + (matrix_A[i2 * MAT_SIZE + k2 + 11] * matrix_B[(k2 + 11) * MAT_SIZE + j2])
                                                    + (matrix_A[i2 * MAT_SIZE + k2 + 12] * matrix_B[(k2 + 12) * MAT_SIZE + j2])
                                                    + (matrix_A[i2 * MAT_SIZE + k2 + 13] * matrix_B[(k2 + 13) * MAT_SIZE + j2])
                                                    + (matrix_A[i2 * MAT_SIZE + k2 + 14] * matrix_B[(k2 + 14) * MAT_SIZE + j2])
                                                    + (matrix_A[i2 * MAT_SIZE + k2 + 15] * matrix_B[(k2 + 15) * MAT_SIZE + j2]);
            }
          }
        }
      }
    }
  }
}

void* parallel_matmul_opt(void* thread_id) {
  long t = (long)thread_id;
  long section = MAT_SIZE/NUM_THREADS;

  for(int i = t*section; i<((t+1)*section); i+=block_size) {
    for(int j = 0; j<MAT_SIZE; j+=block_size) {
      for(int k = 0; k<MAT_SIZE; k+=block_size) {
        for(int i2 = i; i2<i+block_size; ++i2) {
          for(int j2 = j; j2<j+block_size; ++j2) {
            for(int k2 = k; k2<k+block_size; k2+=16) {
              parallel_opt_C[i2 * MAT_SIZE + j2] += (matrix_A[i2 * MAT_SIZE + k2] * matrix_B[k2 * MAT_SIZE + j2]) 
                                                    + (matrix_A[i2 * MAT_SIZE + k2 + 1] * matrix_B[(k2 + 1) * MAT_SIZE + j2])
                                                    + (matrix_A[i2 * MAT_SIZE + k2 + 2] * matrix_B[(k2 + 2) * MAT_SIZE + j2])
                                                    + (matrix_A[i2 * MAT_SIZE + k2 + 3] * matrix_B[(k2 + 3) * MAT_SIZE + j2])
                                                    + (matrix_A[i2 * MAT_SIZE + k2 + 4] * matrix_B[(k2 + 4) * MAT_SIZE + j2])
                                                    + (matrix_A[i2 * MAT_SIZE + k2 + 5] * matrix_B[(k2 + 5) * MAT_SIZE + j2])
                                                    + (matrix_A[i2 * MAT_SIZE + k2 + 6] * matrix_B[(k2 + 6) * MAT_SIZE + j2])
                                                    + (matrix_A[i2 * MAT_SIZE + k2 + 7] * matrix_B[(k2 + 7) * MAT_SIZE + j2])
                                                    + (matrix_A[i2 * MAT_SIZE + k2 + 8] * matrix_B[(k2 + 8) * MAT_SIZE + j2])
                                                    + (matrix_A[i2 * MAT_SIZE + k2 + 9] * matrix_B[(k2 + 9) * MAT_SIZE + j2])
                                                    + (matrix_A[i2 * MAT_SIZE + k2 + 10] * matrix_B[(k2 + 10) * MAT_SIZE + j2])
                                                    + (matrix_A[i2 * MAT_SIZE + k2 + 11] * matrix_B[(k2 + 11) * MAT_SIZE + j2])
                                                    + (matrix_A[i2 * MAT_SIZE + k2 + 12] * matrix_B[(k2 + 12) * MAT_SIZE + j2])
                                                    + (matrix_A[i2 * MAT_SIZE + k2 + 13] * matrix_B[(k2 + 13) * MAT_SIZE + j2])
                                                    + (matrix_A[i2 * MAT_SIZE + k2 + 14] * matrix_B[(k2 + 14) * MAT_SIZE + j2])
                                                    + (matrix_A[i2 * MAT_SIZE + k2 + 15] * matrix_B[(k2 + 15) * MAT_SIZE + j2]);
            }
          }
        }
      }
    }
  }
  return 0;
}

double rtclock() {
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday(&Tp, &Tzp);
  if (stat != 0) {
    printf("Error return from gettimeofday: %d\n", stat);
  }
  return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void check_result(uint64_t* first_res, uint64_t* second_res) {
  double maxdiff, this_diff;
  int numdiffs;
  int i, j;
  numdiffs = 0;
  maxdiff = 0;

  for (i = 0; i < MAT_SIZE; i++) {
    for (j = 0; j < MAT_SIZE; j++) {
      this_diff = first_res[i * MAT_SIZE + j] - second_res[i * MAT_SIZE + j];
      if (this_diff < 0)
        this_diff = -1.0 * this_diff;
      if (this_diff > THRESHOLD) {
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

int main(int argc, char* argv[]) {

  if(argc < 3) {
    cout<<"Usage ./problem5 <number_of_threads> <number_of_blocks> \n";
    return -1;
  }

  NUM_THREADS = atoi(argv[1]);
  block_size = atoi(argv[2]);

  matrix_A = new (std::nothrow) uint64_t[MAT_SIZE * MAT_SIZE];
  matrix_B = new (std::nothrow) uint64_t[MAT_SIZE * MAT_SIZE];
  sequential_C = new (std::nothrow) uint64_t[MAT_SIZE * MAT_SIZE];
  sequential_opt_C = new (std::nothrow) uint64_t[MAT_SIZE * MAT_SIZE];
  parallel_C = new (std::nothrow) uint64_t[MAT_SIZE * MAT_SIZE];
  parallel_opt_C = new (std::nothrow) uint64_t[MAT_SIZE * MAT_SIZE];

  for (int i = 0; i < MAT_SIZE; i++) {
    for (int j = 0; j < MAT_SIZE; j++) {
      matrix_A[(i * MAT_SIZE) + j] = 1;
      matrix_B[(i * MAT_SIZE) + j] = 1;
      sequential_C[(i * MAT_SIZE) + j] = 0;
      sequential_opt_C[(i * MAT_SIZE) + j] = 0;
      parallel_C[(i * MAT_SIZE) + j] = 0;
      parallel_opt_C[(i * MAT_SIZE) + j] = 0;
    }
  }
  pthread_t thread_arr[NUM_THREADS];

  double clkbegin, clkend;

  clkbegin = rtclock();
  sequential_matmul();
  clkend = rtclock();
  cout << "Time for Sequential version: " << (clkend - clkbegin) << "seconds.\n";

  clkbegin = rtclock();
  for (int i = 0; i < NUM_THREADS; i++) {
    pthread_create(&thread_arr[i], NULL, parallel_matmul, (void*)(intptr_t)i);
  }

  for (int i = 0; i < NUM_THREADS; i++) {
    pthread_join(thread_arr[i], NULL);
  }

  clkend = rtclock();
  cout << "Time for parallel version: " << (clkend - clkbegin) << "seconds.\n";


  pthread_t thread_arr1[NUM_THREADS];

  clkbegin = rtclock();
  sequential_matmul_opt();
  clkend = rtclock();
  cout << "Time for Optimised Sequential version: " << (clkend - clkbegin) << "seconds.\n";

  clkbegin = rtclock();
  for (int i = 0; i < NUM_THREADS; i++) {
    pthread_create(&thread_arr1[i], NULL, parallel_matmul_opt, (void*)(intptr_t)i);
  }

  for (int i = 0; i < NUM_THREADS; i++) {
    pthread_join(thread_arr1[i], NULL);
  }

  clkend = rtclock();
  cout << "Time for Optimised parallel version: " << (clkend - clkbegin) << "seconds.\n";

  check_result(sequential_C, parallel_C);
  check_result(sequential_C, sequential_opt_C);
  check_result(sequential_C, parallel_opt_C);
  return 0;
}
