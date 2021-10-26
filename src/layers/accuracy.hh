#include <tensor.hh>

float find_accuracy(Tensor<int> &labels, Tensor<float> &predictions){
  labels.debugTensor();
  predictions.debugTensor();
  labels.viewTensor();
  predictions.viewTensor();
  int noClasses = predictions.dim2;
  int examples = labels.dim1;
  int acc = 0;
  for(int i=0;i<labels.dim1;i++){
    bool correct = true;
    int pred_class = labels.data_host[i];
    float pred_score = predictions.data_host[i*noClasses + pred_class];
    for(int j=0;j<predictions.dim2;j++){
        if(j != pred_class){
          if(predictions.data_host[i*noClasses + j] > pred_score){
            correct = false;
            break;
          }
        }
    }
    if(correct)acc ++;
  }
  // free(labels.data_host);
  float final_acc = (1.0 * acc) / examples;

  return final_acc;
}
