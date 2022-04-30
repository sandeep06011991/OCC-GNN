#include<iostream>
class obj{

public:
  int tid;
  obj(int tid){
    this->tid = tid;
    std::cout << "Constructor on " << this->tid <<"\n";
  }

  ~obj(){
    std::cout << "Destructor on " << this->tid << "\n";
  }
};

int main(){
  {
    obj a(1);

  }
  std::cout << "hello world\n";
}
