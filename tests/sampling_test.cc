#include <iostream>
#include <string>
#include "dataset.h"
#include <fstream>
#include "tensor.hh"
#include "samplers/full.h"
// #include "gnn/sage.hh"

int main(){

  std::string BIN_DIR = "/mnt/homes/spolisetty/data/tests/gcn";
  Dataset * dataset = new Dataset(BIN_DIR + "/pubmed");
  std::cout << dataset->num_edges <<"\n";
  std::cout << dataset->fsize << "\n";
std::cout << " classes: " << dataset->noClasses << "\n";
  auto *s = new TwoHopNoSample(dataset->num_nodes, dataset->num_edges,
               dataset->indptr, dataset->indices, 1000);

  s->shuffle();
    // Test 1 print out all samples.
  int noB = s->number_of_batches();
  std::cout << "number of batches is" << noB <<"\n";
  for(int bid = 0; bid<noB;bid++){
    s->get_sample(bid);
  }


  // Test 2 print out all csr neighbourhoods.


  std::cout << "Hello World ! \n";
}
