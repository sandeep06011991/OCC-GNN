#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "graph.h"
#include <vector>
#include "dataset.h"
#include "tensor.hh"

using namespace std;

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}

// one layer gcn
// void distributed_gcn(Graph *sample){
//     vector<ComputeGraph *> graphs = get_list_of_compute_graphs(sample);

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

// }

int main() {
    // Move this into input variable later.
    string graph_dir = "/mnt/homes/spolisetty/data/occ/pubmed";
    Dataset * dataset = new Dataset(graph_dir);

    Tensor<float> *F = new Tensor<float>(dataset->features, dataset->num_nodes, dataset->fsize);
    Tensor<int> *Labels = new Tensor<int>(dataset->labels, dataset->num_nodes, 1);
    // Initialize data on GPU
    // Initialize model weights on gpu
    // Forward Pass.
    // Compute Loss
    // Backward Pass
    // Combine everything into training.

    std::cout << "Hello world!\n";
    return 0;
}
