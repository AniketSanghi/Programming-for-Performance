// Compile: g++ -std=c++11 find-max.cpp -o find-max -ltbb

#include <cassert>
#include <chrono>
#include <iostream>
#include </usr/include/tbb/blocked_range.h>
#include </usr/include/tbb/parallel_reduce.h>

using std::cout;
using std::endl;
using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;

#define N (1 << 26)

uint32_t serial_find_max(const uint32_t* a) {
  uint32_t value_of_max = 0;
  uint32_t index_of_max = 0;
  for (uint32_t i = 0; i < N; i++) {
    uint32_t value = a[i];
    if (value > value_of_max) {
      value_of_max = value;
      index_of_max = i;
    }
  }
  return index_of_max;
}

// Implement a parallel max function with Intel TBB
class FindMax {
  const uint32_t* my_a;
public:
  uint32_t max_val;
  uint32_t index;

  FindMax(const uint32_t* a): my_a(a), max_val(0), index(0) {}
  FindMax(FindMax &x, tbb::split): my_a(x.my_a), max_val(0), index(0) {}

  void operator()(const tbb::blocked_range<uint32_t>& r) {
    int curr_max_val = max_val;
    uint32_t curr_index = index;
    for(uint32_t i = r.begin(); i != r.end(); ++i) {
      uint32_t value = my_a[i];
      if (value > curr_max_val) {
        curr_max_val = value;
        curr_index = i;
      }
    }
    max_val = curr_max_val;
    index = curr_index;
  }

  void join(const FindMax &y) {
    if(max_val < y.max_val || (max_val == y.max_val && index > y.index)) {
      max_val = y.max_val;
      index = y.index;
    }
  }
};

uint32_t tbb_find_max(const uint32_t* a) {
  FindMax find_max(a);
  tbb::parallel_reduce(tbb::blocked_range<uint32_t>(0,N), find_max);
  return (find_max.index);
}

int main() {
  uint32_t* a = new uint32_t[N];
  for (uint32_t i = 0; i < N; i++) {
    a[i] = i;
  }

  HRTimer start = HR::now();
  uint64_t s_max_idx = serial_find_max(a);
  HRTimer end = HR::now();
  auto duration = duration_cast<microseconds>(end - start).count();
  cout << "Sequential max index: " << s_max_idx << " in " << duration << " us" << endl;

  start = HR::now();
  uint64_t tbb_max_idx = tbb_find_max(a);
  end = HR::now();
  assert(s_max_idx == tbb_max_idx);
  duration = duration_cast<microseconds>(end - start).count();
  cout << "Parallel (TBB) max index in " << duration << " us" << endl;

  return EXIT_SUCCESS;
}
