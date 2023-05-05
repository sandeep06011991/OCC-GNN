#include <stdio.h>
__global__
void test(){
	int j = 0;
	j = threadIdx.x;
	printf("check");
}

int main(){
	int i = 0;
	i = i + 1;
	cudaSetDevice(0);
	test<<<1,2>>>();
	cudaDeviceSynchronize();

}
