#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

/*
# copy, fill and sequence
- Initialize all elements of a vector to a specific value
- Copy certain elements from one to another
- - thrust::copy function can be used to copy a range of host or device elements to another host or device vector
- - thrust::fill sets a range of elements to a specific value 
- - thrust::sequence can be used to create a sequence of equally spaced values
*/

int main (void) {
  // initialize all ten integers of device_vector to 1
  thrust::device_vector<int> D(10, 1);

  // set the first seven elements of vetor to 9
  thrust::fill(D.begin(), D.begin() + 7, 9);

  // initialize host vector with the first 5 elements of D
  thrust::host_vector<int> H(D.begin(), D.begin() + 5);

  // set the elements of H to 0, 1, 2, 3..
  thrust::sequence(H.begin(), H.end());

  // copy of all of H back to the beginning of D
  thrust::copy(H.begin(), H.end(), D.begin());

  // print D
  for (int i=0; i<D.size(); i++)
    std::cout << "[>] D[" << i << "] = " << D[i] << std::endl;

  return 0;
}
