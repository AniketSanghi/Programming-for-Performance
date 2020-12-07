// Compile: g++ -std=c++11 -fopenmp fibonacci.cpp -o fibonacci -ltbb

#include <cassert>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <tbb/tbb.h>

#define N 50

using std::cout;
using std::endl;
using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;

// Serial Fibonacci
long ser_fib(int n) {
  if (n == 0 || n == 1) {
    return (n);
  }
  return ser_fib(n - 1) + ser_fib(n - 2);
}

// Implement OpenMP version with explicit tasks
long omp_fib_v1_aux(int n) {
  if (n == 0 || n == 1) {
    return (n);
  }

  long a,b;

  #pragma omp task untied shared(a)
  a = omp_fib_v1_aux(n-1);

  #pragma omp task untied shared(b)
  b = omp_fib_v1_aux(n-2);

  #pragma omp taskwait
  return (a+b);
}

long omp_fib_v1(int n) {
  if (n == 0 || n == 1) {
    return (n);
  }

  long fib;

  #pragma omp parallel num_threads(12)
  {
    #pragma omp single
    {
      fib = omp_fib_v1_aux(n);
    }
  }

  return fib;
}

// Implement an optimized OpenMP version with any valid optimization
long omp_fib_v2_aux(int n) {
  if (n == 0 || n == 1) {
    return (n);
  }

  if (n < 25) {
    return omp_fib_v2_aux(n-1) + omp_fib_v2_aux(n-2);
  }

  long a,b;

  #pragma omp task untied shared(a)
  a = omp_fib_v2_aux(n-1);

  #pragma omp task untied shared(b)
  b = omp_fib_v2_aux(n-2);

  #pragma omp taskwait
  return (a+b);
}

long omp_fib_v2(int n) {
  if (n == 0 || n == 1) {
    return (n);
  }

  long fib;

  #pragma omp parallel num_threads(12)
  {
    #pragma omp single
    {
      fib = omp_fib_v2_aux(n);
    }
  }

  return fib;
}

// Implement Intel TBB version with blocking style
class FibTask_1: public tbb::task {
public:
  const int n;
  long* const fib;
  FibTask_1(int n_, long* fib_): n(n_), fib(fib_) {}

  tbb::task* execute() {
    // Set a appropriate cutoff for N
    if(n < 25) {
      *fib = ser_fib(n);
    } else {
      long f1,f2;

      FibTask_1 &t1 = *new(allocate_child())FibTask_1(n-1, &f1);
      FibTask_1 &t2 = *new(allocate_child())FibTask_1(n-2, &f2);
      set_ref_count(3);
      spawn(t1);
      spawn_and_wait_for_all(t2);
      *fib = f1 + f2;
    }
    return NULL;
  }
};

long tbb_fib_blocking(int n) {
  long fib = 0;
  FibTask_1 &task1 = *new(tbb::task::allocate_root())FibTask_1(n, &fib);
  tbb::task::spawn_root_and_wait(task1);
  return fib;
}

// Implement Intel TBB version with continuation passing style
class FibC: public tbb::task {
public:
  long* const fib;
  long f1, f2;
  FibC(long *fib_): fib(fib_) {}

  tbb::task* execute() {
    *fib = f1 + f2;
    return NULL;
  }
};

class FibTask_2: public tbb::task {
public:
  const int n;
  long* const fib;
  FibTask_2(int n_, long* fib_): n(n_), fib(fib_) {}

  tbb::task* execute() {
    // Set a appropriate cutoff for N
    if(n < 25) {
      *fib = ser_fib(n);
    } else {
      FibC &final = *new(allocate_continuation())FibC(fib);
      FibTask_2 &t1 = *new(final.allocate_child())FibTask_2(n-1, &final.f1);
      FibTask_2 &t2 = *new(final.allocate_child())FibTask_2(n-2, &final.f2);
      final.set_ref_count(2);
      spawn(t1);
      spawn(t2);
    }
    return NULL;
  }
};

long tbb_fib_cps(int n) {
  long fib = 0;
  FibTask_2 &task1 = *new(tbb::task::allocate_root())FibTask_2(n, &fib);
  tbb::task::spawn_root_and_wait(task1);
  return fib;
}

int main(int argc, char** argv) {
  cout << std::fixed << std::setprecision(5);

  HRTimer start = HR::now();
  long s_fib = ser_fib(N);
  HRTimer end = HR::now();
  auto duration = duration_cast<microseconds>(end - start).count();
  cout << "Serial time: " << duration << " microseconds" << endl;

  start = HR::now();
  long omp_v1 = omp_fib_v1(N);
  end = HR::now();
  assert(s_fib == omp_v1);
  duration = duration_cast<microseconds>(end - start).count();
  cout << "OMP v1 time: " << duration << " microseconds" << endl;

  start = HR::now();
  long omp_v2 = omp_fib_v2(N);
  end = HR::now();
  assert(s_fib == omp_v2);
  duration = duration_cast<microseconds>(end - start).count();
  cout << "OMP v2 time: " << duration << " microseconds" << endl;

  start = HR::now();
  long blocking_fib = tbb_fib_blocking(N);
  end = HR::now();
  assert(s_fib == blocking_fib);
  duration = duration_cast<microseconds>(end - start).count();
  cout << "TBB (blocking) time: " << duration << " microseconds" << endl;

  start = HR::now();
  long cps_fib = tbb_fib_cps(N);
  end = HR::now();
  assert(s_fib == cps_fib);
  duration = duration_cast<microseconds>(end - start).count();
  cout << "TBB (cps) time: " << duration << " microseconds" << endl;

  return EXIT_SUCCESS;
}
