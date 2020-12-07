#include <iostream>
#include <string.h>
#include <pthread.h>

using namespace std;

const int N = 4096;
int T = 1;
int Bl = 1;
uint64_t A[N][N], B[N][N], C[N][N];

void single_matrix_multiply() {
	for(int i = 0; i<N; i+=Bl) {
		for(int j = 0; j<N; j+=Bl) {
			for(int k = 0; k<N; k+=Bl) {
				for(int i2 = i; i2<i+Bl; ++i2) {
					for(int j2 = j; j2<j+Bl; ++j2) {
						for(int k2 = k; k2<k+Bl; ++k2) {
							C[i2][j2] += A[i2][k2] * B[k2][j2];
						}
					}
				}
			}
		}
	}
}

void *parallelised_matrix_multiply(void *token) {
	long t = (long)token;
	int section = N/T;

	for(int i = t*section; i<((t+1)*section); i+=Bl) {
		for(int j = 0; j<N; j+=Bl) {
			for(int k = 0; k<N; k+=Bl) {
				for(int i2 = i; i2<i+Bl; ++i2) {
					for(int j2 = j; j2<j+Bl; ++j2) {
						for(int k2 = k; k2<k+Bl; ++k2) {
							C[i2][j2] += A[i2][k2] * B[k2][j2];
						}
					}
				}
			}
		}
	}
}

void single_matrix_multiply4() {
	for(int i = 0; i<N; i+=Bl) {
		for(int j = 0; j<N; j+=Bl) {
			for(int k = 0; k<N; k+=Bl) {
				for(int i2 = i; i2<i+Bl; ++i2) {
					for(int j2 = j; j2<j+Bl; ++j2) {
						for(int k2 = k; k2<k+Bl; k2+=4) {
							C[i2][j2] += A[i2][k2] * B[k2][j2];
							C[i2][j2] += A[i2][k2+1] * B[k2+1][j2];
							C[i2][j2] += A[i2][k2+2] * B[k2+2][j2];
							C[i2][j2] += A[i2][k2+3] * B[k2+3][j2];
						}
					}
				}
			}
		}
	}
}

void *parallelised_matrix_multiply4(void *token) {
	long t = (long)token;
	int section = N/T;

	for(int i = t*section; i<((t+1)*section); i+=Bl) {
		for(int j = 0; j<N; j+=Bl) {
			for(int k = 0; k<N; k+=Bl) {
				for(int i2 = i; i2<i+Bl; ++i2) {
					for(int j2 = j; j2<j+Bl; ++j2) {
						for(int k2 = k; k2<k+Bl; k2+=4) {
							C[i2][j2] += A[i2][k2] * B[k2][j2];
							C[i2][j2] += A[i2][k2+1] * B[k2+1][j2];
							C[i2][j2] += A[i2][k2+2] * B[k2+2][j2];
							C[i2][j2] += A[i2][k2+3] * B[k2+3][j2];
						}
					}
				}
			}
		}
	}
}

void single_matrix_multiply8() {
	for(int i = 0; i<N; i+=Bl) {
		for(int j = 0; j<N; j+=Bl) {
			for(int k = 0; k<N; k+=Bl) {
				for(int i2 = i; i2<i+Bl; ++i2) {
					for(int j2 = j; j2<j+Bl; ++j2) {
						for(int k2 = k; k2<k+Bl; k2+=8) {
							C[i2][j2] += A[i2][k2] * B[k2][j2];
							C[i2][j2] += A[i2][k2+1] * B[k2+1][j2];
							C[i2][j2] += A[i2][k2+2] * B[k2+2][j2];
							C[i2][j2] += A[i2][k2+3] * B[k2+3][j2];
							C[i2][j2] += A[i2][k2+4] * B[k2+4][j2];
							C[i2][j2] += A[i2][k2+5] * B[k2+5][j2];
							C[i2][j2] += A[i2][k2+6] * B[k2+6][j2];
							C[i2][j2] += A[i2][k2+7] * B[k2+7][j2];
						}
					}
				}
			}
		}
	}
}

void *parallelised_matrix_multiply8(void *token) {
	long t = (long)token;
	int section = N/T;

	for(int i = t*section; i<((t+1)*section); i+=Bl) {
		for(int j = 0; j<N; j+=Bl) {
			for(int k = 0; k<N; k+=Bl) {
				for(int i2 = i; i2<i+Bl; ++i2) {
					for(int j2 = j; j2<j+Bl; ++j2) {
						for(int k2 = k; k2<k+Bl; k2+=8) {
							C[i2][j2] += A[i2][k2] * B[k2][j2];
							C[i2][j2] += A[i2][k2+1] * B[k2+1][j2];
							C[i2][j2] += A[i2][k2+2] * B[k2+2][j2];
							C[i2][j2] += A[i2][k2+3] * B[k2+3][j2];
							C[i2][j2] += A[i2][k2+4] * B[k2+4][j2];
							C[i2][j2] += A[i2][k2+5] * B[k2+5][j2];
							C[i2][j2] += A[i2][k2+6] * B[k2+6][j2];
							C[i2][j2] += A[i2][k2+7] * B[k2+7][j2];
						}
					}
				}
			}
		}
	}
}

int main(int argc, char* argv[]) {

	if(argc < 4) {
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
	Bl = stoi(argv[2]);
	int U = stoi(argv[3]);

	if(T == 1) {
		if(U == 1) single_matrix_multiply();
		else if(U == 4) single_matrix_multiply4();
		else if(U == 8) single_matrix_multiply8();
	}
	else {

		pthread_t thread[T];
		
		if(U == 1) {
			for(long i = 0; i<T; ++i) {
				int ret = pthread_create(&thread[i], NULL, parallelised_matrix_multiply, (void *)i);
				if(ret) {
					cout<<"Error in creating threads\n";
					exit(-1);
				}
			}
		} else if(U == 4) {
			for(long i = 0; i<T; ++i) {
				int ret = pthread_create(&thread[i], NULL, parallelised_matrix_multiply4, (void *)i);
				if(ret) {
					cout<<"Error in creating threads\n";
					exit(-1);
				}
			}
		} else if(U == 8) {
			for(long i = 0; i<T; ++i) {
				int ret = pthread_create(&thread[i], NULL, parallelised_matrix_multiply8, (void *)i);
				if(ret) {
					cout<<"Error in creating threads\n";
					exit(-1);
				}
			}
		}

		for(int i = 0; i<T; ++i) {
			pthread_join(thread[i], NULL);
		}
	}

	return 0;
}