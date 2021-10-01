//Runtime eorrs: Runtime errors occur while the program is running.
//For example : Divide by zero,File not found,out of memmory
//They can crash the program. Exception handling can help deal with runtume errors.




#include <iostream>
using namespace std; 

int main()
{
    int x,y;
    float z;
    x = 0;
    y = 0;
    z = x/y;
    cout << z ;//Nothing gwts printed out. Gets skipped.

    return 0;

}

int zero(){
    int num;

    num = 0/0;

    cout<<num;//Error thrown is : division by zero [-Wdiv-by-zero]  num = 0/0;

    return 0;
}