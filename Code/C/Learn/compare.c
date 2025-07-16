#include <stdio.h>

int max(int num1, int num2){
    return num1 > num2 ? num1 : num2;
}

int main(void){
    int num1, num2;

    printf("Enter two numbers: ");
    scanf("%d %d", &num1, &num2);

    int num3 = max(num1, num2);

    printf("%d", num3);
}
