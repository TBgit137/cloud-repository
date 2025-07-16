#include <stdio.h>

#define PI 3.14

int add(int a, int b);

int main(){
    int num1, num2, sum;

    printf("Enter two numbers: ");
    scanf("%d %d", &num1, &num2);

    sum = add(num1, num2);

    printf("Sum of %d and %d is %d\n", num1, num2, sum);

    return 0;
    //return
}

int add(int a, int b){
    return a + b;
}