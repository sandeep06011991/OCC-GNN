
class Layer{

    virtual Tensor& computeForwardPass(Tensor& in);
  	virtual Tensor& computeBackPass(Tensor& dW);

};

class Sequential{

public:

  Layers **layers;
  int no_layers;

  Sequential(no_layers);

};
