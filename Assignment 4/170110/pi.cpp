// g++ -std=c++11 -fopenmp pi.cpp -o pi -ltbb

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <tbb/tbb.h>

using std::cout;
using std::endl;
using std::chrono::duration_cast;
using HR = std::chrono::high_resolution_clock;
using HRTimer = HR::time_point;
using std::chrono::microseconds;

const int NUM_INTERVALS = std::numeric_limits<int>::max();

double serial_pi() {
  double dx = 1.0 / NUM_INTERVALS;
  double sum = 0.0;
  for (int i = 0; i < NUM_INTERVALS; ++i) {
    double x = (i + 0.5) * dx;
    double h = std::sqrt(1 - x * x);
    sum += h * dx;
  }
  double pi = 4 * sum;
  return pi;
}

// Implement OpenMP version with minimal false sharing
double omp_pi() {
  double dx = 1.0 / NUM_INTERVALS;
  double sum = 0.0;

  #pragma omp parallel for reduction(+:sum) schedule(dynamic, 1000000)
  for (int i = 0; i < NUM_INTERVALS; ++i) {
    double x = (i + 0.5) * dx;
    double h = std::sqrt(1 - x * x);
    sum += h * dx;
  }
  double pi = 4 * sum;
  return pi;
}

// Implement TBB version with parallel algorithms
class PiCalc {
  double dx = 1.0 / NUM_INTERVALS;
public:
  double sum;

  PiCalc(): sum(0.0f) {}
  PiCalc(PiCalc &x, tbb::split): sum(0.0f) {}

  void operator()(const tbb::blocked_range<int>& r) {
    double sum_calc = sum;
    for(int i = r.begin(); i != r.end(); ++i) {
      double x = (i + 0.5) * dx;
      double h = std::sqrt(1 - x * x);
      sum_calc += h * dx;
    }
    sum = sum_calc;
  }

  void join(const PiCalc &y) {
    sum += y.sum;
  }
};

double tbb_pi() {
  PiCalc pi;
  tbb::parallel_reduce(tbb::blocked_range<int>(0, NUM_INTERVALS), pi);
  return (4*pi.sum);
}

int main() {
  cout << std::fixed << std::setprecision(5);

  HRTimer start = HR::now();
  double ser_pi = serial_pi();
  HRTimer end = HR::now();
  auto duration = duration_cast<microseconds>(end - start).count();
  cout << "Serial pi: " << ser_pi << " Serial time: " << duration << " microseconds" << endl;

  start = HR::now();
  double o_pi = omp_pi();
  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  cout << "Parallel pi (OMP): " << o_pi << " Parallel time: " << duration << " microseconds"
       << endl;

  start = HR::now();
  double t_pi = tbb_pi();
  end = HR::now();
  duration = duration_cast<microseconds>(end - start).count();
  cout << "Parallel pi (TBB): " << t_pi << " Parallel time: " << duration << " microseconds"
       << endl;

  return EXIT_SUCCESS;
}

// Local Variables:
// compile-command: "g++ -std=c++11 pi.cpp -o pi; ./pi"
// End:
