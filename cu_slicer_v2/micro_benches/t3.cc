#include<iostream>
class A{
	public:
	int i;
	
	A(int i){
		this->i = i;
	}
	
	A(){}
	
	~A(){
		std::cout << "destructor\n";		
	}
};


void func(A &a){
	A b = a;
	std::cout << "hello \n";
}

int main(){
	A b;
	b.i = 1;
	func(b);
}	
