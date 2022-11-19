#include <iostream>
#include <iomanip>
#include <cmath>
#include <thrust/random.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>

// the example suggests to vary the values M and N to find a "sweet spot"
__host__ __device__ int hash(unsigned int a) {

  a = (a+0x7ed55d16) + (a<<12);
  a = (a^0xc761c23c) ^ (a>>19);
  a = (a+0x165567b1) + (a<<5);
  a = (a+0xd3a2646c) ^ (a<<9);
  a = (a+0xfd7046c5) + (a<<3);
  a = (a^0xb55a4f09) ^ (a>>16);
  return a;
}

struct estimate_pi : public thrust::unary_function<unsigned int, float> {
  __host__ __device__ float operator()(unsigned int thread_id) {
    float sum = 0;
    unsigned int N = 10000;  // the nummber of samples per thread
    unsigned int seed = hash(thread_id);

    // seed random number
    thrust::default_random_engine rng(seed);

    // create mapping for random numbers from [0-1]
    thrust::uniform_real_distribution<float> u01(0,1);

    // take N samples in a quarter circle
    for(unsigned int i=0; i<N; ++i) {
      // draw a sample from the unit square 
      float x = u01(rng);
      float y = u01(rng);
      // distance from origin
      float distance = sqrtf(x*x + y*y);

      // add 1.0f if (u0,u1) is inside the quater circle
      if (distance <= 1.0f) sum += 1.0f;
    }    
    // multiply by four to get the area of the whole circle
    sum *= 4.0f;

    // divide by N
    return sum / N;
  }
};

int main (void) {
  // use 30k independant seeds
  int M = 30000;

  float estimate = thrust::transform_reduce(
    // create iterators
    thrust::counting_iterator<int>(0), 
    thrust::counting_iterator<int>(M), 
    estimate_pi(),
    0.0f, 
    thrust::plus<float>()
  );
  estimate /= M;
  std::cout << std::setprecision(6);
  std::cout << "[+] pi is approximately " << estimate << std::endl;

  return 0;
}
