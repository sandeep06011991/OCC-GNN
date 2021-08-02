#include<vector>
#ifndef GRAPH_H
#define GRAPH_H

using namespace std;
// Store graph in csr format.
// cpu structure which contains global information of cpu.
class Graph {
  int no_vertices;
  int no_edges;
  // neighbour size of node i = offset[i+1] - offset[i]
  // len(offsets) = no_vertices
  int *offsets;
  // neighbours of node i range from values[]
  // len(values) = no_edges
  int *values;
  // Each node F in the graph is mapped to a gpu_id
  // len(gpu_id) == no_vertices
  int *gpu_id;
  // Each node gets a new id local to the gpu id;
  int *local_gpu_id;

  Graph(int no_Vertices,int no_edges);
} ;

struct NodePartition{
  int gpu_id;
  int no_nodes;
  int feature_size;
  // Mapping from gpu_local to global id;
  int * local_to_global;
  // access features[local_id] refers to local_to_global[local_id]
  // in the global graph
  double * features;
};

// Estimate size of input and output.
struct ComputeGraph{
  // If src==dest: Compute and keep in the same gpu.
  // If src != dest: Compute and merge at destination.
  int src_gpu_id;
  int dst_gpu_id;
  int no_vertices;
  int no_edges;
  int feature_size;

  int *ids;
  int *offsets;
  int *edges ;

  // src-gpu pointers
  int *src_id_d;
  int *src_offsets_d;
  int *src_edges_d;
  int *src_out_d;

  // dest-gpu pointers
  int *dest_id_d;
  int *dest_out_d;

  void  gpu_allocate_memory(ComputeGraph * graph);
  void  compute(ComputeGraph * graph);
  void  move(ComputeGraph * graph);
  void  merge(ComputeGraph * graph);
  void  de_allocate(ComputeGraph * graph);

};


vector<ComputeGraph *> get_list_of_compute_graphs(Graph *sample);

Graph * get_toy_graph();

Graph * get_default_sample();

void distribute_graph(Graph *graph, int gpus[]);

#endif 
