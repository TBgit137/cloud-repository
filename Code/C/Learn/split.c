#include <stdio.h>

int main(void){
    printf("Enter a tree digit number: ");

    int num;

    scanf("%d", &num);

    int num1 = num / 100;

    int num2 = (num % 100) / 10;

    int num3 = num % 10;

    printf("%d\n%d\n%d\n", num1, num2, num3);
}
