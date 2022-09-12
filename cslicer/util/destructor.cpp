#include<iostream>
#include <memory>
class obj{

public:
  int tid;
  obj(){}

  obj(int tid){
    this->tid = tid;
    std::cout << "Constructor on " << this->tid <<"\n";
  }

  ~obj(){
    std::cout << "Destructor on " << this->tid << "\n";
  }

  obj operator=(const obj& b) {
        obj box;
        std::cout << "There is a copy\n";
        return box;
     }
};
// Note Free 2 is never called.

void func1(obj &obj){

}

void func2(obj obj){

}

int main(){
  {
    obj a(1);
    obj *b = new obj(2);
    auto c = std::make_shared<obj>(3);
    // auto d = c;
    obj &d = *b;
    func1(a);
    func2(*b);
  }
  std::cout << "hello world\n";
}
