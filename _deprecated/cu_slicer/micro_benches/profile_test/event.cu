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
   thrust::device_vector<int> nodes = v1;
    cudaEvent_t event1, event2;
	cudaEventCreate(&event1);
	cudaEventCreate(&event2);
    cudaEventRecord(event1,0);
    thrust::sort(nodes.begin(), nodes.end());
    auto it = thrust::unique(nodes.begin(), nodes.end());
    nodes.erase(it, nodes.end());
    cudaEventRecord(event2,0);
    cudaEventSynchronize(event2);
    float time;
    cudaEventElapsedTime(&time, event1,event2);
    nvtxRangePushA(__FUNCTION__ ":nvtxRangePushA");
nvtxRangePop(); 

   std::cout <<"end\n";
}
