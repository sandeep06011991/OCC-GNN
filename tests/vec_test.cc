

#include<vector>
#include<iostream>

using namespace std;

void foo(const vector<int>& b){
	b[0]= 20;
	std::cout << b[0] << "\n";
	return;
}


int main(){

	vector<int> a;
	a.push_back(10);
	foo(a);
	std::cout << a[0] <<"\n";
}
