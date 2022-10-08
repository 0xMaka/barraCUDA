#include <iostream>
#include <math.h>

// cpp template - cuda prep

// function to add the elements of two arrays
void add(int n, float *x, float *y) {
  for (int i=0; i<n; i++)
		y[i] = x[i] + y[i];
}
int main(void) {
	int N = 1<<20; //20mill elements
	
	float *x = new float[N];
	float *y = new float[N];
  
	// initialize x and y arrays on the host (cpu)
	for (int i=0; i<N; i++) {
    x[i] = 1.0f;
		y[i] = 2.0f;
	}

	// run kernel on 1mill elements of the cpu
	add(N, x, y);
	
	// check for errors (all values should be 3.0f)
  float maxError = 0.0f;
	for (int i; i<N; i++) 
		maxError = fmax(maxError, fabs(y[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;

	// free memory
	delete [] x;
	delete [] y;
}
