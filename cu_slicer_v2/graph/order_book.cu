
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <fstream>
#include "order_book.h"
using namespace std;
__global__
void test(OrderBook * od){
    for(int i = 0 ; i < 5; i++){
        printf("%d \n",od->partition_offsets[i]);
    }
    for(int i = 0 ; i < 4; i++){
        for(int j = 0 ; j < 4; j++){
        printf("%d \n",od->cached_offsets[i][j]);
    }
    }
}

vector<int> split(string str, string token){
    vector<int>result;
    while(str.size()){
        int index = str.find(token);
        if(index!=string::npos){
            result.push_back(stoi(str.substr(0,index)));
            str = str.substr(index+token.size());
            if(str.size()==0)result.push_back(stoi(str));
        }else{
            result.push_back(stoi(str));
            str = "";
        }
    }
    return result;
}



OrderBook::OrderBook(std::string BIN_DIR, std::string graphname,\
                 std::string size, int partitions){
        {   
            assert(partitions < MAX_GPUS);
            this->num_partitions = partitions;    
            {std::fstream file(BIN_DIR + "/" + graphname +"/order_book_"+ size + ".txt",std::ios::in);
            std::string line;
            int partition = 0;
            while(getline(file,line)){
                auto v = split(line, ",");
                assert(v.size() == partitions);
                for(int j = 0; j < partitions ; j ++ ){
                   cached_offsets[partition][j] = v[j]; 
                }
                partition ++;
            
            }
            assert(partition == partitions);
                 
        }
        {
        std::fstream file(BIN_DIR + "/" + graphname + "/partition_offsets.txt");
        std::string line;
        getline(file,line);
        auto v = split(line, ",");
        assert(v.size() == partitions + 1);
            for(int i = 0; i < partitions+1; i++){
                partition_offsets[i] = v[i];
           }
        }
        gpuErrchk(cudaMalloc(&order_book_d, sizeof(OrderBook)));
        gpuErrchk(cudaMemcpy(order_book_d, this, sizeof(OrderBook), cudaMemcpyHostToDevice));
        test<<<1,1>>>(order_book_d);
        gpuErrchk(cudaDeviceSynchronize());
    }

};

