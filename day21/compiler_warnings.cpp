//Compiler Warnings: Compiler warnings are warnings thrown by the compiler regarding the program to let know of any errors. 
//Warnings,unike Compiler errors do not interrupt the compilation process.

#include <iostream>

int main()
{
    int fav_num;
    int num;

    num = 100;

    std::cout << "Hello World!\n";
    std::cout << fav_num; //error C4700 : uninitialized local variable 'fav_num' used

}
