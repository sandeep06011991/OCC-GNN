#include<iostream>
#include<stdio.h>


// Merge Point
// Requirements [a_start, a_end], [b_start, b_end]
// size a = [a_end - a_start + 1]
// Given 2 arrays of size [a_start, a_end] b. have total elements (a+b)
// Goal break the work into (a+b)/2 elements
// Takes in size s = len(a) + len(b) and returns merge point a,b  a+b= s/2
// if  = 3, returns such, idx a,b that a+b = 1
// Both inclusive [a_start, a_end]
std::tuple<int,int> print_merge(int *a, int a_start, int a_end,
      int *b, int b_start, int b_end){
    int a_start = 0;
    int a_end = a_size;
    int b_start = 0;
    int b_end = b_size;
    int t = (a_size + b_size)/2;
    while(a_start < a_end & b_start < b_end){
      int mid1 = (a_start + a_end)/2;
      int mid2 = (b_start + b_end)/2
      if(a[mid1] > b[mid2 + 1]) and (a[mid1] < b[mid2]){
        return std::make_tuple<int,int>(mid1, mid2);
      }
      if(a[mid1] < b[mid2 + 1]){
          // Shift
      }else{
          //Shift
      }
    }
}

void test_correctness(int *a, int a_size, int *b, int b_size){
  x,y = print_merge(a, a_size, b, b_size);
  // Goal
  (x + 1) + (y + 1) == (a_size + b_size)/2;
  // Merge[0,x][0,y],[x+1,x][y+1,y]
  if (x+1 != a_size){
    a[x+1] < b[y];
  }
  if ()
}

int main(){
  // Test cases.
  // a is empty, b is empty.
  // a is odd, b is even
  // a is odd, b is odd
  // a is even, b is even.
  // Test correctness.
  std::vector<int> a{1,2,3,4,5,6,7};
  std::vector<int> b{3,4,5,6,8,9,10};
  print_merge(a.data(),a.size(),b.data(),b.size());
  std::cout << "hello world\n";
}
