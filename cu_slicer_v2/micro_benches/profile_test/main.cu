#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <iostream>
int main(){
	thrust::host_vector<int> v1;
   cudaSetDevice(0);
   for(int i=100; i >0 ;  i--){
   	v1.push_back(i);
   }
   std::cout << v1.size();
   thrust::device_vector<int> v2 = v1;

   thrust::sort(thrust::device, v2.begin(), v2.end()); 

   std::cout <<"end\n";
}

