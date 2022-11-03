#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

/*
# host_vector and device_vector 
- - host_vector is stored in CPU memory, device_vector in GPU device memory
- Just like std::vector, they are generic containers 
- - Able to store any data type
- - an be resized dynamically. 
*/

void printd(thrust::device_vector<int> &D);

int main (void) {

  std::cout << "[x] ================================================================" << std::endl;
  std::cout << "[-] Using thrust::host_vector to initialize an array of 4 integers.." << std::endl;
  // H has storage for 4 integers
  thrust::host_vector<int> H(4);

  std::cout << "[-] Assigning values to elements through standard bracket notation.." << std::endl;
  //initialize individual elements
  H[0] = 14;
  H[1] = 20;
  H[2] = 38;
  H[3] = 46;
  
  std::cout << "[-] Calling .size() method to return the length of the array." << std::endl;
  std::cout << "[>] H has a size of: " << H.size() << std::endl;
  std::cout << "[>] "; 
  for (int i=0; i<H.size(); i++) {
    std::cout << "H[" << i << "] = " << H[i];
    if (i < H.size() - 1)
      std::cout << ", "; 
  }
  std::cout << std::endl;
  
  std::cout << "[x] ----------------------------------------------------------------" << std::endl;
  std::cout << "[-] Using .resize(n) to dynamically modify the array's length.." << std::endl;
  
  // resize H
  H.resize(2);
  std::cout << "[>] H now has a size of: " << H.size() << std::endl;

  std::cout << "[x] ----------------------------------------------------------------" << std::endl;
  std::cout << "[-] Initializing a device array, assigning it the contents of H." << std::endl;
  // copy host_vector H to device_vector D
  thrust::device_vector<int> D = H;
  printd(D);

  std::cout << "[-] Modifying D's values with standard bracket assignment." << std::endl;
  //elements of D can be modified
  D[0] = 99;
  D[1] = 88;
  printd(D); 
  
  std::cout << "[-] H and D are auto deleted when function returns." << std::endl;
  std::cout << "[x] ===============================================================" << std::endl;
  return 0;
}

void printd(thrust::device_vector<int> &D ) {
  // print contents of D
  for (int i=0; i<D.size(); i++)
    std::cout << "[>] D[" << i << "] = " << D[i] << std::endl;
}
