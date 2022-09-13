#include <iostream>
#include "dataset.h"
#include "models/model1.hh"
#include "models/model2.hh"
#include "layers/relu.hh"
#include "samplers/full.h"
#include <assert.h>
#include "layers/accuracy.hh"

int main(){

  std::string BIN_DIR = "/mnt/homes/spolisetty/data/tests/gcn";
  Dataset * dataset = new Dataset(BIN_DIR + "/pubmed");
  std::cout << dataset->num_edges <<"\n";
  std::cout << dataset->fsize << "\n";
  std::cout << " classes: " << dataset->noClasses << "\n";
  auto *s = new TwoHopNoSampler(dataset->num_nodes, dataset->num_edges,
               dataset->indptr, dataset->indices, 1024,dataset->features,
                dataset->labels, dataset->fsize);

  auto  model = new GCNBasic(dataset->fsize, 256, dataset->noClasses);

  int noB = s->number_of_batches();
  // int noB = 1;
  s->shuffle();
  std::cout << "number of batches is" << noB <<"\n";

  CrossEntropyLoss* loss = new CrossEntropyLoss();

  int noEpochs = 1;

  for(int i=0;i<noEpochs;i++){
    for(int bid = 0; bid<noB;bid++){
      std::cout << "epoch "<< i << "batch " << bid <<"\n";
      s->get_sample(bid);
      size_t free, total;
      cudaMemGetInfo( &free, &total );
      std::cout << "GPU  memory: free=" << free << ", total=" << total << std::endl;

    // Create tensor data from samples.
    SampleLayer &l2 = s->sample.l2;
    Tensor<int> * indptr1 = new Tensor<int>(l2.indptr.data(),l2.indptr.size(),1);
    Tensor<int> * indices1 = new Tensor<int>(l2.indices.data(),l2.indices.size(),1);
    Tensor<float> * batch_in = new Tensor<float>(s->batch_features,l2.in_nodes,dataset->fsize);
    SampleLayer &l1 = s->sample.l1;
    Tensor<int> * indptr2 = new Tensor<int>(l1.indptr.data(),l1.indptr.size(),1);
    Tensor<int> * indices2 = new Tensor<int>(l1.indices.data(),l1.indices.size(),1);
    Tensor<int> * batch_labels = new Tensor<int>(s->batch_labels, l1.out_nodes, 1);


    Tensor<float>& out = model->forward(*indptr1, *indices1, *indptr2, *indices2, \
          *batch_in , *batch_labels);

    Tensor<float>& total_loss = loss->forward(out, *batch_labels);
    Tensor<float> & grad_out = loss->backward(*batch_labels);



    model->backward(grad_out);


    float s = model->ll1->dW->debugTensor();
    // model->ll1->dW->viewTensor();
    if (s!=s){
      // Print everything
      std::cout << "Print everything\n";
      std::cout << "out " << out.debugTensor();
      out.viewTensor();
      std::cout << "loss " << total_loss.debugTensor() << "\n";
      total_loss.viewTensor();
      batch_labels->debugTensor();
      std::cout << "grad_out " << grad_out.debugTensor();
      for(int i=0;i<grad_out.dim1 * grad_out.dim2;i++){
        float tt = grad_out.data_host[i];
        if(tt != tt){
          int row = i/grad_out.dim2;
          int col = i/grad_out.dim1;
          std::cout << "Total output";
          for(int k=0;k<grad_out.dim2;k++){
              std::cout << out.data_host[row*grad_out.dim2 + k] << " ";
          }
          std::cout << "Correct label" << batch_labels->  data_host[row] <<"\n";
          std::cout << "Not a number " << i  <<" : "<< out.data_host[i] <<"\n";
        }
      }
      grad_out.viewTensor();
      std::cout << "Model L2" << model->ll2->dW->debugTensor();
      model->ll2->dW->viewTensor();
      std::cout << "GCN" << model->gcn2->out_grad->debugTensor();
      model->gcn2->out_grad->viewTensor();
      std::cout << "RELU" << model->relu->d_out->debugTensor();
      model->relu->d_out->viewTensor();

    }
    float s1 = model->ll1->W->debugTensor();
    model->ll1->W->viewTensor();
    assert(s1 == s1);
    assert(s == s);

    model->update(.1);

    // float loss_val = out.debugTensor();
    // assert(loss_val == loss_val);
    // float total_loss_val = total_loss.debugTensor();
    // assert(total_loss_val == total_loss_val);
    // float grad_out_val = grad_out.debugTensor();
    // assert(grad_out_val == grad_out_val);

    //   std::cout << "Calculate loss \n";
    float loss = total_loss.debugTensor();
    std::cout << "printing loss " << total_loss.debugTensor()/total_loss.dim1 <<" \n ";
    float acc = find_accuracy(*batch_labels, out);
    std::cout << "accuracy is (expect close to one):" <<  acc <<" xx \n";
    // if(loss != loss){
    //   assert(false);
    // }
    // delete batch_in;
    // model->backward(out);


    indptr1->clearTensor();
    indices1->clearTensor();
    batch_in->clearTensor();

    indptr2->clearTensor();
    indices2->clearTensor();
    batch_labels->clearTensor();
    delete indptr1, indices1, batch_in, indptr2, indices2, batch_labels;

  }
}

  std::cout << "Test to show I can train 200 epochs !!\n";
  // Test 2 sample add one backward pass
  // Add loop code while also deleting tensors.
}
