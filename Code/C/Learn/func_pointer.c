#include <stdio.h>

int add(int, int);
int minus(int, int);
int mul(int, int);
int div(int, int);

int main() {
    int (*arr[4])(int, int) = {add, minus, mul, div};

    int calc(int num1, int num2, int operator) {
        return (arr[operator - 1])(num1, num2);
    }

    int num1, num2, operator;

    printf("enter two numbers and operator: ");
    scanf("%d %d %d", &num1, &num2, &operator);

    printf("%d\n", calc(num1, num2, operator));
}

int add(int num1, int num2) {
    return num1 + num2;
}

int minus(int num1, int num2) {
    return num1 - num2;
}

int mul(int num1, int num2) {
    return num1 * num2;
}

int div(int num1, int num2) {
    if(num2 == 0) {
        printf("Error: Division by zero\n");
        return 0;
    }
    return num1 / num2;
}
