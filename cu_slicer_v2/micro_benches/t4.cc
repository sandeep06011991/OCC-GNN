#include<iostream>
#include<vector>
using namespace std;
int main()
{
    int arr[] = {1, 3, 4, 7, 8, 9};
    // Create a vector from array
    vector<int> numbers(arr, arr + sizeof(arr) / sizeof(int));
    // Print all elements of vector on console
    for(const auto & elem: numbers) {
        cout<<elem<< ", ";
    }
    std::cout << sizeof(arr) <<"\n"; 
    return 0;
}
