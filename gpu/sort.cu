#include<iostream>
#include<chrono>

__global__ void duplicates_single_thread_gpu
(int *in, int in_size, int *out, int *out_size){
  int prev = in[0];
  int t = 0;
  for(int i=1;i<in_size;i++){
    if(prev!=in[i]){
      out[t] = prev;
      t ++;
      prev = in[i];
    }
  }
  out[t] = prev;
  t++;
  *out_size = t;
}

void cpu_single_thread
        (int *in, int in_size, int *out){
  int prev = in[0];
  int t = 0;
  for(int i=1;i<in_size;i++){
    if(prev!=in[i]){
      out[t] = prev;
      t ++;
      prev = in[i];
    }
  }
  out[t] = prev;
  t++;
  // *out_size = t;
}

void fancy_merge_gpu(){}

int main(){
  cudaSetDevice(0);
  std::cout << "hello world\n";
  int arr_size = 1000;
  int bucket = 10;
  int *arr = (int *)malloc(arr_size *sizeof(int));
  int *out = (int *)malloc(arr_size *sizeof(int));
  for(int i=0;i<arr_size;i++){
    arr[i] = i/bucket;
  }
  auto start = std::chrono::system_clock::now();
  cpu_single_thread(arr,arr_size, out);
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Elapsed time cpu: " << elapsed.count() << "s\n";

  int *in_d, *in_s_d, *out_d, *out_s_d;
  cudaMalloc(&in_d, arr_size *sizeof(int));
  cudaMalloc(&out_d, arr_size *sizeof(int));
  cudaMalloc(&out_s_d, sizeof(int));
  auto start00 = std::chrono::system_clock::now();
  cudaMemcpy(in_d,arr,arr_size*sizeof(int),cudaMemcpyHostToDevice);
  auto start1 = std::chrono::system_clock::now();
  duplicates_single_thread_gpu<<<1,1>>>(in_d,arr_size,out_d,out_s_d);
  cudaDeviceSynchronize();
  auto end1 = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed1 = end1 - start1;
  std::cout << "Elapsed time gpu: " << elapsed1.count() << "s\n";
  std::chrono::duration<double> elapsed2 = start1 - start00;
  std::cout << "Elapsed time gpu movement: " << elapsed2.count() << "s\n";

}
