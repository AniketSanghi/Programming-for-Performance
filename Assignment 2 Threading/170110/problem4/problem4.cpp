#include <iostream>
#include <pthread.h>
#include <string.h>
#include <queue>
#include <unistd.h>

using namespace std;

pthread_mutex_t lock_FIFO = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond_FIFO = PTHREAD_COND_INITIALIZER;

pthread_mutex_t lock_customer_wake = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond_customer_wake[24];
pthread_mutex_t lock_customer_wake2 = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond_customer_wake2[24];
pthread_mutex_t lock_teller_wake = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t cond_teller_wake[2];
pthread_mutex_t lock_print = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t lock_token = PTHREAD_MUTEX_INITIALIZER;

int N;
queue <int> customer_queue;
long counter = 1;

int assigned_teller[24];
int token_for_teller[2];
int number_of_customers_serviced = 0;
int custId[2];
long tokens[24];

int shared_array[16];
pthread_mutex_t lock_elem[16];

void *customer_exec(void* value) {
	long val = (long)value;
	long teller_ = -1;

	// Assign token
	pthread_mutex_lock(&lock_token);
		tokens[val] = counter;
		counter++;
	pthread_mutex_unlock(&lock_token);

	// Enter the Queue, Synchronized
	pthread_mutex_lock(&lock_FIFO);
		customer_queue.push(val);
		pthread_cond_signal(&cond_FIFO);
	pthread_mutex_unlock(&lock_FIFO);

	// Wait till any teller is free to help you
	pthread_mutex_lock(&lock_customer_wake);
		while(assigned_teller[val] == -1) {
			pthread_cond_wait(&cond_customer_wake[val], &lock_customer_wake);
		}
		teller_ = assigned_teller[val];
	pthread_mutex_unlock(&lock_customer_wake);

	// Enter data into the shared array
	int count = 0;
	for(int i = 0; i<16; ++i) {
		pthread_mutex_lock(&lock_elem[i]);
			if(shared_array[i] == -1) {
				shared_array[i] = tokens[val];
				count++;
			}
			if(count == 8) {
				pthread_mutex_unlock(&lock_elem[i]);
				break;
			}
		pthread_mutex_unlock(&lock_elem[i]);
	}

	// Wake the teller, tell him to process the entered data
	pthread_mutex_lock(&lock_teller_wake);
		token_for_teller[teller_] = val;
		pthread_cond_signal(&cond_teller_wake[teller_]);
	pthread_mutex_unlock(&lock_teller_wake);

	// Wait for the tellers signal about the completion of transaction
	pthread_mutex_lock(&lock_customer_wake2);
		while(token_for_teller[teller_] != -1) {
			pthread_cond_wait(&cond_customer_wake2[val], &lock_customer_wake2);
		}
	pthread_mutex_unlock(&lock_customer_wake2);

	// Customer informs the next person in the queue before leaving
	pthread_mutex_lock(&lock_FIFO);
		if(!customer_queue.empty()) {
			custId[teller_] = customer_queue.front();
			customer_queue.pop();
			number_of_customers_serviced += 1;
		} else {
			custId[teller_] = -1;
		}
	pthread_mutex_unlock(&lock_FIFO);
	return 0;
}

void *teller_exec(void* token) {
	long val = (long)token;
	long cust_id = -1;

	// Handle case of 1 customer where only 1 teller is needed
	if(N == 1 && val == 1) return 0;

	// Wait till any customer enters the queue
	pthread_mutex_lock(&lock_FIFO);
		while(customer_queue.empty() && number_of_customers_serviced < N) {
			pthread_cond_wait(&cond_FIFO, &lock_FIFO);
		}
		custId[val] = customer_queue.front();
		customer_queue.pop();
		number_of_customers_serviced += 1;
	pthread_mutex_unlock(&lock_FIFO);


	while(true) {

		// The customer, the teller will assist
		cust_id = custId[val];

		// When all customers have been assisted
		if(cust_id == -1) break;

		// Wait till the customer writes into the array
		pthread_mutex_lock(&lock_teller_wake);
			token_for_teller[val] = -1;
			pthread_mutex_lock(&lock_customer_wake);
				assigned_teller[cust_id] = val;
				pthread_cond_signal(&cond_customer_wake[cust_id]);
			pthread_mutex_unlock(&lock_customer_wake);
			while(token_for_teller[val] == -1) {
				pthread_cond_wait(&cond_teller_wake[val], &lock_teller_wake);
			}
		pthread_mutex_unlock(&lock_teller_wake);

		// Read and print the array
		pthread_mutex_lock(&lock_print);
			for(int i = 0; i<16; ++i) {
				pthread_mutex_lock(&lock_elem[i]);
					if(shared_array[i] == tokens[token_for_teller[val]]) {
						cout<<shared_array[i]<<" ";
						shared_array[i] = -1;
					}
				pthread_mutex_unlock(&lock_elem[i]);
			}
			cout<<"\n";
		pthread_mutex_unlock(&lock_print);

		// Signal the customer that the transaction is complete
		pthread_mutex_lock(&lock_customer_wake2);
			token_for_teller[val] = -1;
			pthread_cond_signal(&cond_customer_wake2[cust_id]);
		pthread_mutex_unlock(&lock_customer_wake2);

		// Sleep for 5 seconds
		sleep(5);
	}
	return 0;
}

int main(int argc, char* argv[]) {

	if(argc < 2) {
		cout<<"Please pass the number of customers as command line arguments\n";
		exit(-1);
	}

	N = stoi(argv[1]);
	if(N == 0) return 0;

	// Initialisation
	for(int i = 0; i<N; ++i) {
		int ret = pthread_cond_init(&cond_customer_wake[i], NULL);
		if(ret) {
			cout<<"Cond Initialisation Failed\n";
			exit(-1);
		}
		ret = pthread_cond_init(&cond_customer_wake2[i], NULL);
		if(ret) {
			cout<<"Cond Initialisation Failed\n";
			exit(-1);
		}
		assigned_teller[i] = -1;
	}
	for(int i = 0; i<2; ++i) {
		int ret = pthread_cond_init(&cond_teller_wake[i], NULL);
		if(ret) {
			cout<<"Cond Initialisation Failed\n";
			exit(-1);
		}
	}
	for(int i = 0; i<16; ++i) {
		int ret = pthread_mutex_init(&lock_elem[i], NULL);
		if(ret) {
			cout<<"Lock Initialisation Failed\n";
			exit(-1);
		}
		shared_array[i] = -1;
	}

	pthread_t teller[2];
	pthread_t customer[N];

	pthread_attr_t attr;

    /* Initialize and set thread detached attribute */
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	for(int i = 0; i<N; ++i) {
		int ret = pthread_create(&customer[i], &attr, customer_exec, (void *)(intptr_t)i);
		if(ret) {
			cout<<"Error in creating customer thread\n";
			exit(-1);
		}
	}
	for(int i = 0; i<2; ++i) {
		int ret = pthread_create(&teller[i], &attr, teller_exec, (void *)(intptr_t)i);
		if(ret) {
			cout<<"Error in creating teller thread\n";
			exit(-1);
		} 
	}

	for(int i = 0; i<N; ++i) pthread_join(customer[i], NULL);
	for(int i = 0; i<2; ++i) pthread_join(teller[i], NULL);

	return 0;
}