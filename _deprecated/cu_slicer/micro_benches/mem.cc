#include<iostream>
class A{
public:
  A() = default;
    A(A const&) = default;

       	A(){

 }

 ~A(){
	 std::cout << "Free";} 


  A& operator=(const A& b){

	  std::cout <<"Equals"; 
	  return (*this);
  }

};

class B{

	A a;
 public:
	A& getA(){
	return a;
	}
	
	
};
int main(){
	B b;
	A d;
	A a = b.getA();
	A c = d;

}
