#include <iostream>
#include <queue>
#include <mutex>
#include <thread>
#include <condition_variable>
#include "util/conqueue.h"



class obj{
public:
  int a;
  obj(int a){
    this->a = a;
  }
};

void producer_thread(ConQueue<obj *> *a, int tid){
  // std::cout << "producer " << tid <<"\n";
  for(int i=0;i<3;i++){
    std::cout << "push " << tid + i<<"\n";
    a->push_object(new obj(tid + i));
  }
}

void consumer_thread(ConQueue<obj *> *a){
  for(int j=0;j<9;j++){
    std::cout << (a->pop_object()->a) <<"\n";
  }
}

// Test Queue with back pressure.
int test_conqueue(){
  ConQueue<obj *> *a = new ConQueue<obj *>(3);
  int threads = 3;
  std::thread **th = (std::thread **)malloc(sizeof(std::thread) * threads);
  std::cout << "starting threads\n";
  for(int i=0;i<threads;i++){
    th[i] = new std::thread(producer_thread,a,(i+1)*10);
  }
  consumer_thread(a);
  std::cout << "All threads started\n";
  for(int i=0;i<threads;i++){
    th[i]->join();
  }

  std::cout << "Hello world\n";
}

int main(){
  test_conqueue();
}
// #include "sample.h"
// template class ConQueue<std::vector<long> *>;
// template class ConQueue<Sample *>;
