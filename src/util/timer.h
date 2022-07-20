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
    SAMPL_F_DATA_TRANSFER,
    MOVEMENT_COST,
    MOVEMENT_COMPUTE1,
    MOVEMENT_COMPUTE2,
    FILL_DATA,
    CREATE_CSR,
    DUPLICATE_LAYER,
    DEBUG1,
    DEBUG2,
    DEBUG3
};

extern float TIMERS[1000];
extern bool ACTIVE[1000];
extern std::chrono::time_point<std::chrono::high_resolution_clock> START[1000];

void active_timer(measure m);

inline void start_timer(measure m){
  START[m] =  high_resolution_clock::now();
}

inline void stop_timer(measure m){
    auto stop = high_resolution_clock::now();
    auto duration = ((float)duration_cast<microseconds>(stop - START[m]).count())/1000000;
    // if(m == MOVEMENT_COST){
	    // std::cout << duration << "\n";
    // }
    TIMERS[m] += duration;
}

inline void add_timer_ms(measure m,float f){
   TIMERS[m] += f/1000;
}
void reset_timers();

void print_timer();
