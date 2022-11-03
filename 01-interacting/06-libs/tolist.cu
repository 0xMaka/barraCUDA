#include <list>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/copy.h>

int main (void) {

  std::list<int> stl_list;
  stl_list.push_back(10);
  stl_list.push_back(20);
  stl_list.push_back(30);
  stl_list.push_back(40);

  thrust::device_vector<int> D(stl_list.begin(), stl_list.end());
  std::vector<int> stl_vector(D.size());
  thrust::copy(D.begin(), D.end(), stl_vector.begin());

  return 0;
}

