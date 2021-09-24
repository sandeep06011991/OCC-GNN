#pragma once
#include "layers/linear.hh"

void train_single_gpu(Linear *lh,Tensor * in){
    Tensor * next = in;
    for(int i=0;i<no_layers;i++){
        next = lh[i]->forward_pass(next);
    }
    // Compute loss
    Tensor * gradients;
    for(int i=no_layers-1;i <=0; i--){
        gradients = lh[i]->backward_pass(gradients);
        lh[i]->update();
    }
}
