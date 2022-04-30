#pragma once
#include <queue>
#include <iostream>
#include <queue>
#include <mutex>
#include <thread>
#include <condition_variable>

template
<typename T>
class ConQueue{
  int max_size = 0;
  std::queue<T> queue;
  std::mutex mtx;
  std::condition_variable has_space;
  std::condition_variable not_empty;

public:
  ConQueue(int max_size);

  void push_object(T  object);

  T pop_object();

};
