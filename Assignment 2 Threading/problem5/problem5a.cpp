#include <iostream>
#include <string.h>
#include <pthread.h>

using namespace std;

const int N = 4096;
int T = 1;
uint64_t A[N][N], B[N][N], C[N][N];

void single_matrix_multiply() {
	for(int i = 0; i<N; ++i) {
		for(int j = 0; j<N; ++j) {
			for(int k = 0; k<N; ++k) {
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
}

void *parallelised_matrix_multiply(void *token) {
	long t = (long)token;
	int section = N/T;

	for(int i = t*section; i<((t+1)*section); ++i) {
		for(int j = 0; j<N; ++j) {
			for(int k = 0; k<N; ++k) {
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
}

int main(int argc, char* argv[]) {

	if(argc < 2) {
		cout<<"Please pass the number of threads as command line arguments (1 for no threads)\n";
		exit(-1);
	}

	// Initialisation
	for(int i = 0; i<N; ++i) {
		for(int j = 0; j<N; ++j) {
			A[i][j] = B[i][j] = 1;
			C[i][j] = 0;
		}
	}

	T = stoi(argv[1]);

	if(T == 1) single_matrix_multiply();
	else {

		pthread_t thread[T];
		for(long i = 0; i<T; ++i) {
			int ret = pthread_create(&thread[i], NULL, parallelised_matrix_multiply, (void *)i);
			if(ret) {
				cout<<"Error in creating threads\n";
				exit(-1);
			}
		}
		for(int i = 0; i<T; ++i) {
			pthread_join(thread[i], NULL);
		}
	}

	return 0;
}