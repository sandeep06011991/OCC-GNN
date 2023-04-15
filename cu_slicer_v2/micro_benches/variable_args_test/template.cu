#include<iostream>
int ADD(int a, int b){
	return (a + b);
}



template<typename F, typename ...args>
void print(F, args ...a){
	std::cout << "IO STR" <<F(a...);
}


int main(){
	//Test templating.
	//
	//
	print<int(*)(int,int), int, int>(ADD, 10,20);
}
