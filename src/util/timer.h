#pragma once
#include <chrono>
#include <iostream>
using namespace std;
using namespace std::chrono;

enum measure{
    TIME1,
    TIME2,
    SAMPLE_CREATION,
    SAMPL_F_DATA_FORMAT,
    SAMPL_F_DATA_TRANSFER
};

float TIMERS[1000];
bool ACTIVE[1000];
std::chrono::time_point<std::chrono::high_resolution_clock> START[1000];

void active_timer(measure m){
  ACTIVE[m] = true;
}

inline void start_timer(measure m){
  START[m] =  high_resolution_clock::now();
}

inline void stop_timer(measure m){
    auto stop = high_resolution_clock::now();
    auto duration = (float)duration_cast<milliseconds>(stop - START[m]).count()/1000;
    TIMERS[m] += duration;
}

void reset_timers(){
  for(int i=0;i<1000;i++){
    TIMERS[i] = 0;
    ACTIVE[i] = false;
  }
}

void print_timer(){
  for(int i=0;i<1000;i++){
    if(ACTIVE[i]){
      switch(i){
        case TIME1: std::cout << "time 1";
                    break;
        case TIME2: std::cout << "time 2";
                    break;
        case SAMPLE_CREATION: std::cout << "Sample Creation |";
                    break;
        case SAMPL_F_DATA_FORMAT: std::cout << "Data Format Creation|";
                      break;
        case SAMPL_F_DATA_TRANSFER:std::cout << "Data Format Transfer|";
                    break;
      }
      std::cout << TIMERS[i] <<"s\n";
    }
  }
}
