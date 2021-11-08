#include<iostream>
// Check that I can do a p2p copy 
float p2p_copy (size_t size)
{
  int *pointers[2];

  size = sizeof(int);
  cudaSetDevice (0);
  cudaDeviceEnablePeerAccess (1, 0);
  cudaMalloc (&pointers[0], size);
  int t = 1991;
  cudaMemcpy(pointers[0], &t, size ,cudaMemcpyHostToDevice);

  cudaSetDevice (1);
  cudaDeviceEnablePeerAccess (0, 1);
  cudaMalloc (&pointers[1], size);

  cudaEvent_t begin, end;
  cudaEventCreate (&begin);
  cudaEventCreate (&end);

  cudaEventRecord (begin);
  cudaMemcpy (pointers[1], pointers[0],size, cudaMemcpyDeviceToDevice);
  cudaEventRecord (end);
  int c;
  cudaMemcpy(&c,pointers[1],sizeof(int), cudaMemcpyDeviceToHost);
  cudaEventSynchronize (end);

  float elapsed;
  cudaEventElapsedTime (&elapsed, begin, end);
  elapsed /= 1000;

  cudaSetDevice (0);
  cudaFree (pointers[0]);

  cudaSetDevice (1);
  cudaFree (pointers[1]);

  std::cout << c <<"\n";
  cudaEventDestroy (end);
  cudaEventDestroy (begin);

  return elapsed;
}

int main(){
  p2p_copy(10);
}
