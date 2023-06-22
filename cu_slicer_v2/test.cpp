struct S{
	int A;
};
#include<iostream>

void sum(int a, int b){}

template<typename... Args>
void functn(int a, Args... args){
	std::cout << "wrapper function called\n";
	sum( a, args...);


}
template<typename f>
void pass_f(f ff,int a, int b){
	ff(a,b);
}

void sum_two(int a,int b, int c){
	
}

class A{

	const int a;
	 A(int b){
	 	a = b; 
	 }
};

void sum(int a, int b);
int main(){
sum(1,2);	
	functn(1,2);
	pass_f(sum, 1 ,2);
}

