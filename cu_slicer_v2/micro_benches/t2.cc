#include<vector>
#include<iostream>

class A{

	A(int a){}

	A(){}
};

int main(){

	int a[] = {1,2,34};
	std::vector<int> b(a, a+3);
	std::vector<int> c = {1,2,3};
	for(int i=0;i<3;i++){
		std::cout << b[i] <<" ";
	}
}
