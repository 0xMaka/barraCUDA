#include <stdlib.h>
#include <string.h>
#include <time.h>

void h_sum_array(float *A, float *B, float *C, const int N) {
  for(int idx=0; idx<N; idx++)
    C[idx] = A[idx] + B[idx];
}

void initial_data(float *ip, int size) {
	time_t t;
	srand((unsigned int) time(&t));
	for (int i=0; i<size; i++) {
    ip[i] = (float) ( rand() &0xFF )/10.0f;
  }
}

int main(int argc, char **argv) {
  int elements = 1024;
	size_t bytes = elements * sizeof(float);
	float *ha, *hb, *hc;
	ha = (float *)malloc(bytes);
	hb = (float *)malloc(bytes);
	hc = (float *)malloc(bytes);

  initial_data(ha, elements);
  initial_data(hb, elements);
  h_sum_array(ha, hb, hc, elements);

	free(ha);
	free(hb);
	free(hc);

	return 0;
}
