#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
using namespace std::chrono;
 
// After function call
namespace py = pybind11;

py::list testlist() {
    std::vector<int> test_vec;
    auto start = high_resolution_clock::now();
    for(int i=0;i<10000 * 100 ;i++){
    	test_vec.push_back(i);
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

// To get the value of duration use the count()
// member function on the duration object
    std::cout << "Construction cost" <<  duration.count() << std::endl;
    return py::cast(test_vec);
}


PYBIND11_MODULE(tensorize, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("test_list", &testlist, "List testing ");
}
