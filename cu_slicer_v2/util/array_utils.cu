#include "device_vector.h"
#include "cuda_utils.h"
#include "array_utils.h"
namespace cuslicer{

// out[tid] = index[in[tid]]
// Used majorly for load balancig.
template<typename T1, typename T2>
void index_in(device_vector<T1>& input, device_vector<T2>& index, device_vector<T2>& out);

template<typename T1>
T1 count_if(device_vector<T1>& input, device_vector<T1>& temp, T1 eq);

};