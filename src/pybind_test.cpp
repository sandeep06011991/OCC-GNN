#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include <vector>
#include "object.h"
namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}


py::list testlist(py::list l) {
    // l.attr("pop")();
    // std::cout << "List has length " << l.size() << std::endl;
    // for (py::handle obj : l) {  // iterators!
    //     std::cout << "  - " << obj.attr("__str__")().cast<std::string>() << std::endl;
    // }
    std::vector<int> test_vec{1, 2, 3, 4, 5};
    py::list test_list3 = py::cast(test_vec);
    l.append(10);  // automatic casting (through templating)!
    return test_list3;
}

PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring
    m.def("test_list", &testlist, "List testing ");
    m.def("add", &add, "A function that adds two numbers");
    py::class_<Pet>(m, "Pet")
        .def(py::init<const std::string &>())
        .def("setName", &Pet::setName)
        .def("getName", &Pet::getName);
}
