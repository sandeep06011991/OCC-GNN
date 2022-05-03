#pragma once
#include <queue>
#include <iostream>
#include <queue>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <vector>

template
<typename T>
class ConQueue{
  int max_size = 0;
  std::queue<T> queue;
  std::mutex mtx;
  std::condition_variable has_space;
  std::condition_variable not_empty;

public:
  ConQueue(int max_size){
      this->max_size = max_size;
  }

  // ConQueue(){}

  void push_object(T  object){
      // Wait for space
      int size = 0;
      while(true){
         std::unique_lock<std::mutex> lck(mtx);
         int size = queue.size();
         if(size < this->max_size){
           queue.push(object);
           not_empty.notify_all();
           break;
         }else{
           has_space.wait(lck);
         }
      }
  }

  T pop_object(){
      T obj;
      std::cout << "attemptiong a pop\n";
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
           not_empty.wait(lck);
         }
        }
      }
      return obj;
  }


};

template class ConQueue<std::vector<long> * >;
