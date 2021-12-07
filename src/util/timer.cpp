#include "util/timer.h"

float TIMERS[1000];
bool ACTIVE[1000];
std::chrono::time_point<std::chrono::high_resolution_clock> START[1000];

void active_timer(measure m){
  ACTIVE[m] = true;
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
        case MOVEMENT_COST:std::cout << "data movement|";
                    break;
        case MOVEMENT_COMPUTE : std::cout << "compute |" ;
                    break;
        case DEBUG1 : std::cout << "debug 1|";
                    break;
        case DEBUG2 : std::cout << "debug 2|";
                    break;
        case DEBUG3 : std::cout << "debug 3|";
                    break;
      }
      std::cout << TIMERS[i] <<"s\n";
    }
  }
}
