#include<time.h>
#include<iostream>
int main(){
	float *a, *b;
	size_t size = (1024 * 1024 * 1024);
	cudaSetDevice(0);
	cudaMalloc(&a, size);
	cudaSetDevice(1);
	cudaMalloc(&b, size);
	float *c;
	cudaMallocHost(&c, size);
	for(int i=0;i<10;i++){
		clock_t begin_time = clock();
		cudaSetDevice(0);
		cudaMemcpy(a,b, size, cudaMemcpyDeviceToDevice);
		cudaDeviceSynchronize();
		float t =  float( clock () - begin_time ) /  CLOCKS_PER_SEC;
		std::cout << "Bandwidth" << 1/t << "GBps\n";
		begin_time = clock();
		cudaMemcpy(a,c, size, cudaMemcpyHostToDevice);
		cudaDeviceSynchronize();
		t =  float( clock () - begin_time ) /  CLOCKS_PER_SEC;
                std::cout << "Host Bandwidth" << 1/t << "GBps\n";

	}

}
