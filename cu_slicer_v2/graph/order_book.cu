
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <fstream>
#include "order_book.h"
using namespace std;


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
            if(partitions != MAX_GPUS){
                std::cout << "Push GPU varaiblity to compile time \n";
            }
            assert(partitions == MAX_GPUS);
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
    }

};

