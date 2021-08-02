#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "graph.h"
#include <vector>


using namespace std;

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

// one layer gcn
void distributed_gcn(Graph *sample){
    vector<ComputeGraph *> graphs = get_list_of_compute_graphs(sample);

    // for(ComputeGraph * graph : graphs){
    //     ComputeGraph::gpu_allocate_memory(graph);
    // }

    // for(ComputeGraph *graph:graphs){
    //     compute(graph);
    // }
    //
    // for(ComputeGraph *graph:graphs){
    //     move(graph);
    // }
    //
    // for(ComputeGraph *graph:graphs){
    //     merge(graph);
    // }
    //
    // for(ComputeGraph *graph:graphs){
    //     de_allocate(graph);
    // }

}

int main() {
    Graph * graph = get_toy_graph();
    cuda_hello<<<1,1>>>();
    cudaDeviceSynchronize();
    std::cout << "Hello world!\n";
    return 0;
}
