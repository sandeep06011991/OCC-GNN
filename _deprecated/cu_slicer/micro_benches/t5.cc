#include<iostream>
class A{
	public:
A& operator=(A const &in){
	std::cout << "copy\n";
	return *this;
}
};

int main(){

	A b;
	A c;
	b = c;
	A d[3];
	d[0] = A();

}
