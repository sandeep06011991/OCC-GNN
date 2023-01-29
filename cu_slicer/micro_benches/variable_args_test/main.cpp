#include <iostream>

int hello(int b){
	std::cout << "I am run:" << b <<"\n";
}

template<typename F, typename...Args>
void outer(F f, Args ...args){
	std::cout << "Outer  is run\n";
	f(args...);
}
int main(){
	const int a = 1;
	#ifdef DEB
		std::cout << "Check\n";
	#endif	
	outer<int(int),int>(hello,1);
}
