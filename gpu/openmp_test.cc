#include<chrono>
#include <iostream>
#include <string>
#include <cstring>
#include <algorithm>

int main(){
  int dim1 = 100024;
  int dim2 = 602;
  int slice_size = dim1 * dim2;

  float * arr = (float *)malloc(sizeof(float) * dim1 * dim2);
  int * slice = (int *)malloc(sizeof(int) * slice_size);
  float * size_out = (float *)malloc(sizeof(float) * dim2);


  auto start = std::chrono::system_clock::now();
#pragma omp parallel for
  for(int i=0; i<slice_size ;i++){
    slice[i] = i*2;
  }
  // std::sort(slice,slice+slice_size);
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Elapsed time1: " << elapsed.count() << "s\n";

  // auto start1 = std::chrono::system_clock::now();
  // slice_two(arr,slice,size_out,dim1, dim2, slice_size);
  // auto end1 = std::chrono::system_clock::now();
  // std::chrono::duration<double> elapsed1 = end1 - start1;
  // std::cout << "Elapsed time2: " << elapsed1.count() << "s\n";

}
