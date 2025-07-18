#include<stdio.h>

union money{
    int money_int;
    float money_float;
    char money_char;
    // same memory space
};

int main(){
    union money m1;
    m1.money_int = 100;
    printf("%d\n", m1.money_int);

    m1.money_float = 100.0;
    printf("%f\n", m1.money_float);

    m1.money_char = 'A';
    printf("%c\n", m1.money_char);
}