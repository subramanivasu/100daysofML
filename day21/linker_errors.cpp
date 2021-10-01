//Liner error : Linker error is throwed when the linker cannot put all the parts of the program together to create an exe but one or more pieces are missing.
//This mostly happens when an obj file or libraries are missing or could not be found.



#include <iostream>

extern int x;//undefined reference to `x'
int main()
{
    int fav_num;
    int num;

    num = 100;

    std::cout << "Hello World!\n";
    std::cout << fav_num; //error C4700 : uninitialized local variable 'fav_num' used
    std::cout << x;

}
