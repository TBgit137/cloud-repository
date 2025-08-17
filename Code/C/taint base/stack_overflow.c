#include <stdio.h>
#include <string.h>

int vuln_fun(char *str);

int main()

{

    char *str = "AAAAAAAAAAAAAAAAAAAAAAAA";

    vuln_fun(str);

    return 0;

}

 

int vuln_fun(char *str)

{

    char stack[10];

    strcpy(stack,str);        // 这里造成溢出！   

}