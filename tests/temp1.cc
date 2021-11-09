#include<iostream>

class a{
  const int *ptr;
public:
  a(){
    ptr = (int *)malloc(1);
  }

  void check(){
    ptr = (int *)malloc(2);
  }
};

int main(){
  a b;
  // b.check();
  int *a =(int *) malloc(0);
  

}
