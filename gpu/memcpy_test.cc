#include<chrono>
#include <iostream>
#include <string>
#include <cstring>

// Check impact of mem:cpy for large dimensions.
void slice_one(float *arr, int *slice, float *size_out,int dim1, int dim2,int  slice_size){
  for(int i=0;i<slice_size;i++){
    for(int j=0;j<dim2;j++){
      size_out[i] = arr[slice[i]*dim2+j];
    }
  }
}


void slice_two(float *arr, int *slice, float *size_out,int dim1, int dim2,int  slice_size){
  for(int i=0;i<slice_size;i++){
    str:memcpy(&size_out[i],&arr[slice[i]*dim2],sizeof(float)*dim2);
  }
}

int main(){
  int dim1 = 100024;
  int dim2 = 602;
  int slice_size = 1024/2;

  float * arr = (float *)malloc(sizeof(float) * dim1 * dim2);
  int * slice = (int *)malloc(sizeof(int) * slice_size);
  float * size_out = (float *)malloc(sizeof(float) * dim2);

  for(int i=0; i<slice_size ;i++){
    slice[i] = i*2;
  }

  auto start = std::chrono::system_clock::now();
  slice_one(arr,slice,size_out,dim1, dim2, slice_size);
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Elapsed time1: " << elapsed.count() << "s\n";

  auto start1 = std::chrono::system_clock::now();
  slice_two(arr,slice,size_out,dim1, dim2, slice_size);
  auto end1 = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed1 = end1 - start1;
  std::cout << "Elapsed time2: " << elapsed1.count() << "s\n";

}
