#include "graph.h"

// A simple toy graph to ease development
// 0 - 1,2,3
// 1 - 3,4
// 2 - 1,4
// 3 -
// 4 - 3
Graph * get_toy_graph(){
    int no_vertices = 5;
    int no_edges = 8;
    Graph * graph = new Graph(no_vertices,no_edges);
    int offset[] = new int[]{0,3,5,7,7};
    int edges[] = new int[]{1,2,3,3,4,1,4,3};
    return graph;
}

Graph * get_default_sample(){
  // This is the whole graph at this point.
  // Not relevant.

}

vector<ComputeGraph *> get_list_of_compute_graphs(Graph *sample){
  return nullptr;

}
