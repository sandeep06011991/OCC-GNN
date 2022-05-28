#include <iostream>
#include <queue>
#include <mutex>
#include <thread>
#include <condition_variable>
#include "util/conqueue.h"

template <typename T>
ConQueue<T>::ConQueue(int max_size){
    this->max_size = max_size;
}



template <typename T>
void ConQueue<T>::push_object(T  object){
    // Wait for space
    int size = 0;
    while(true){
       std::unique_lock<std::mutex> lck(mtx);
       int size = queue.size();
       if(size != this->max_size){
         queue.push(object);
         not_empty.notify_all();
         break;
       }else{
         has_space.wait(lck);
       }
    }
}

template <typename T>
T ConQueue<T>::pop_object(){
    T obj;
    while(true){
      {
       std::unique_lock<std::mutex> lck(mtx);
       int size = queue.size();
       if(size != 0){
         obj = queue.front();
         queue.pop();
         has_space.notify_all();
         break;
       }else{
       std::cout <<"waiting\n";
         not_empty.wait(lck);
       }
      }
    }
    return obj;
}

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

// #include "sample.h"
// template class ConQueue<std::vector<long> *>;
// template class ConQueue<Sample *>;
