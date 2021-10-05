#pragma once

class CSRNeighbourhood{

  int in_nodes;
  int out_nodes;
  vector<int> reorder_id_nd1;
  vector<int> reorder_id_nd2;
  vector<int> offsets;
  vector<int> edges;

  // All samplers use this class to compute csr nieghbourhoods.
  CRNNeighbourhood(Graph * graph, vector<int> nd_1, vector<int> nd_2,
    vector<pair<<int,int> edges){

  }

}
