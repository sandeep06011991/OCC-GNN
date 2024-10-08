#include <iostream>
#include <string>
#include "dataset.h"
#include <fstream>
#include "util/tensor.hh"
#include "samplers/neigh_sample.h"
#include "gnn/sage.hh"
#include "util/timer.h"

// Not a test code runs.
int experiment(string filename){
  reset_timers();
  active_timer(SAMPLE_CREATION);
  active_timer(SAMPL_F_DATA_FORMAT);
  active_timer(SAMPL_F_DATA_TRANSFER);
  std::string BIN_DIR = "/home/spolisetty/data/";
  // Dataset * dataset = new Dataset(BIN_DIR + "/pubmed");
  Dataset * dataset = new Dataset(BIN_DIR + filename);
  // std::cout << dataset->num_edges <<"\n";
  // std::cout << dataset->fsize << "\n";
  // std::cout << " classes: " << dataset->noClasses << "\n";
  int batch_size = 256;
  auto * s = new TwoHopNeighSampler(dataset->num_nodes, dataset->num_edges,
             dataset->indptr, dataset->indices, batch_size,dataset->features,
              dataset->labels, dataset->fsize);

  float * aggr2 = (float *)malloc(sizeof(float) * batch_size * dataset->fsize);
  memset(aggr2, 0, sizeof(float) * batch_size * dataset->fsize);

  // std::cout << "printing neighbourhood \n";
  // int batch_id = 2;
  // for(int k=0;k<batch_size;k++){
  //   int i = batch_id * batch_size + k;
  //   std::cout << i <<":";
  //   for(int nb_offset = dataset->indptr[i]; nb_offset < dataset->indptr[i+1];nb_offset++){
  //     int nb1_id = dataset->indices[nb_offset];
  //     std::cout <<nb1_id <<"(";
  //     for(int nb_offsets1 = dataset->indptr[nb1_id]; nb_offsets1 < dataset->indptr[nb1_id +1];
  //         nb_offsets1++){
  //       int nb2_id = dataset->indices[nb_offsets1];
  //       for(int f = 0;f<dataset->fsize;f++){
  //         aggr2[k*dataset->fsize+f] += dataset->features[nb2_id*dataset->fsize+f];
  //       }
  //       std::cout << nb2_id <<" ";
  //     }
  //     std::cout << ")";
  //   }
  //   std::cout <<"\n";
  // }
  int device_id = 0;
  // Tensor<float> * aggr_t = new Tensor<float>(aggr2, Shape(batch_size, dataset->fsize),device_id);
  // SageAggr * layer1 = new SageAggr(dataset->fsize, device_id);
  // SageAggr * layer2 = new SageAggr(dataset->fsize,device_id);
  //
  s->shuffle();
  int ii = s->number_of_batches();
  for(int i=0;i<ii;i++){
    std::cout << i << ":" << ii <<"\n";
    s->get_sample(i);
    SampleLayer &l2 = s->sample.l2;
    start_timer(SAMPL_F_DATA_TRANSFER);
    Tensor<int> * indptr1 = new Tensor<int>(l2.indptr.data(),Shape(l2.indptr.size(),1),device_id);
    Tensor<int> * indices1 = new Tensor<int>(l2.indices.data(),Shape(l2.indices.size(),1),device_id);
    Tensor<float> * batch_in = new Tensor<float>(s->batch_features,Shape(l2.in_nodes,dataset->fsize),device_id);
    SampleLayer &l1 = s->sample.l1;
    Tensor<int> * indptr2 = new Tensor<int>(l1.indptr.data(),Shape(l1.indptr.size(),1),device_id);
    Tensor<int> * indices2 = new Tensor<int>(l1.indices.data(),Shape(l1.indices.size(),1),device_id);
    Tensor<int> * batch_labels = new Tensor<int>(s->batch_labels, Shape(l1.out_nodes, 1),device_id);
    stop_timer(SAMPL_F_DATA_TRANSFER);
    indptr1->clearTensor();
    indices1->clearTensor();
    batch_in->clearTensor();
    indptr2->clearTensor();
    indices2->clearTensor();
    batch_labels->clearTensor();
  }

  // auto o1 = layer1->forward(*indptr1, *indices1, *batch_in,
  //                 indptr1->s.dim1-1,batch_in->s.dim1);
  // auto o2 = layer2->forward(*indptr2,* indices2, *o1, indptr2->s.dim1-1,o1->s.dim1);
  // std::cout << "total sum" << o2.debugTensor() <<"\n";
  // o2.viewTensor();
  // std::cout << "total sum" << aggr_t->debugTensor() <<"\n";
  // aggr_t->viewTensor();
  //   // Test 1 print out all samples.
  // int noB = s->number_of_batches();
  // // std::cout << "number of batches is" << noB <<"\n";
  // // for(int bid = 0; bid<noB;bid++){
  //   s->get_sample(0);
  // }
  // Test 2 print out all csr neighbourhoods.
  print_timer();

  ofstream file;
  file.open ("exp1.txt",std::ios_base::app | std::ios_base::out);
  file << "ME|" << filename <<"|" <<  2 << "|"<< TIMERS[SAMPLE_CREATION] <<"|" \
      << TIMERS[SAMPL_F_DATA_FORMAT] <<"|" << TIMERS[SAMPL_F_DATA_TRANSFER] <<"\n";
  file.close();
  std::cout << "Hello World ! \n";
  return 1;

}

int main(int argc, char *argv[]){
  std::string filename = (argv[1]);
  experiment(filename);
}
