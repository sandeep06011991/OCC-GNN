#pragma once

// Sample n elements from p elements given a normalized distribution of size p.
// int no. elements.
// Necessary for techniques like ladies.
int sample_n_elements(int *normalized_distribution, int no_elements, int *dest, int target_elements);

bool approx_equal(int a,int b){
  return a==b;
}

bool approx_equal(float a,float b){
  // std::cout << a << " " << b <<"\n";
  // std::cout << "doffer" <<b - a<<"\n";
  if(a==b)return true;
  if(a>b){
    return (a-b)/a < .0001;
  }
  return (b-a)/a < .00001;
}
