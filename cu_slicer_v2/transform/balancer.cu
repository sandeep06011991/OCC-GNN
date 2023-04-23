#pragma once
#include "../util/device_vector.h"

namespace cuslicer{



    // Usually one partition is heavily loaded. 
    class LoadBalancer{

        void balance(device_vector<int> &workload_map, 
            device_vector<int> &sample_in, 
                device_vector<int> &sample_workload_map){

                    // v1 get just indexing done !

                    // v2 sum, find imbalanced partition.
                    // 
                }

    };


};