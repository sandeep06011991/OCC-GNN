#include<iostream>
class cl{
public:
  int a;
  cl(int a){
    this->a = a;
  }
};

class cl1{

  cl * obj = nullptr;
  int i =0;
public:
  cl& invoke(){
    if(obj!=nullptr){
      free(obj);
    }
    obj = new cl(i);
    i++;
    return *obj;
  }
};

int main(){
  std::cout <<"hello world\n";
  cl1 ob;
  ob.invoke();
  for(int i=0;i<1000;i++){
    std::cout << ob.invoke().a << "\n";
  }
}
