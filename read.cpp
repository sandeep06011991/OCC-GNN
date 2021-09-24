#include <iostream>
#include <vector>
#include <string>
#include <fstream>
// #include <fstream>
using namespace std;

int main(){
    string fileName = "meta.json";
    fstream file(fileName,ios::in);
    std::string name;
    std::getline(file, name);
    std::string token = name.substr(name.find("=") + 1,name.length() );
    int i = stoi(token);
    cout << i <<"\n";
    // f

    // string fileName = "test.bin";
    // fstream file(fileName,ios::in|ios::binary);
    // float a[5];
    // file.read((char *)&a,sizeof(a)*5);
    // // file.read((char *)&a,sizeof(a));
    // for(int i=0;i<5;i++){
    //   std::cout << "hello world\n" << a[i] << "read\n";
    // }
}
