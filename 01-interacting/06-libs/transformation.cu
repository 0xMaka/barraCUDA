#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/replace.h>
#include <thrust/functional.h>

void printv(thrust::device_vector<int> &V) {
  for (int i=0; i<V.size(); ++i) {
    std::cout << V[i];
    if (i<V.size() -1)
      std::cout << ", ";
  }
  std::cout << std::endl;
}

void print_all(thrust::device_vector<int> &X, thrust::device_vector<int>&Y, thrust::device_vector<int>&Z) {
  std::cout << "[>] X = ";
  printv(X);
  std::cout << "[>] Y = ";
  printv(Y);
  std::cout << "[>] Z = ";
  printv(Z);
  std::cout << "[x] --------------------------------------------" << std::endl;
}

int main(void) {
  std::cout << "[x] ============================================" << std::endl;
  std::cout << "[-] Allocating 3 device vectors with 10 elements" << std::endl;
  thrust::device_vector<int> X(10);
  thrust::device_vector<int> Y(10);
  thrust::device_vector<int> Z(10);

  print_all(X, Y, Z);

  std::cout << "[-] Initializing X to 0..9" << std::endl;
  thrust::sequence(X.begin(), X.end());
  print_all(X, Y, Z);

  std::cout << "[-] Using thrust::negate to compute Y=-X" << std::endl;
  thrust::transform(X.begin(), X.end(), Y.begin(), thrust::negate<int>());
  print_all(X, Y, Z);

  std::cout << "[-] Using thrust::fill to set Z values to 2" << std::endl;
  thrust::fill(Z.begin(), Z.end(), 2);
  print_all(X, Y, Z);

  std::cout << "[-] Using thrust::modulus to compute Y=X%2" << std::endl;
  thrust::transform(X.begin(), X.end(), Z.begin(), Y.begin(), thrust::modulus<int>());
  print_all(X, Y, Z);

  std::cout << "[-] Using thrust::replace to swap 1->10" << std::endl;
  thrust::replace(Y.begin(), Y.end(), 1, 10);
  print_all(X, Y, Z);

  return 0;
}

/*
[x] ============================================
[-] Allocating 3 device vectors with 10 elements
[>] X = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
[>] Y = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
[>] Z = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
[x] --------------------------------------------
[-] Initializing X to 0..9
[>] X = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
[>] Y = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
[>] Z = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
[x] --------------------------------------------
[-] Using thrust::negate to compute Y=-X
[>] X = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
[>] Y = 0, -1, -2, -3, -4, -5, -6, -7, -8, -9
[>] Z = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
[x] --------------------------------------------
[-] Using thrust::fill to set Z values to 2
[>] X = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
[>] Y = 0, -1, -2, -3, -4, -5, -6, -7, -8, -9
[>] Z = 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
[x] --------------------------------------------
[-] Using thrust::modulus to compute Y=X%2
[>] X = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
[>] Y = 0, 1, 0, 1, 0, 1, 0, 1, 0, 1
[>] Z = 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
[x] --------------------------------------------
[-] Using thrust::replace to swap 1->10
[>] X = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
[>] Y = 0, 10, 0, 10, 0, 10, 0, 10, 0, 10
[>] Z = 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
[x] --------------------------------------------
  */
