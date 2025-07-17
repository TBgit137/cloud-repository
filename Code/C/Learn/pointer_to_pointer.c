#include <stdio.h>

int main() {
    int num = 10;

    int* p1 = &num;

    int** p2 = &p1;

    printf("%p\n", p1);
    printf("%p\n", p2);

    *p2 += 1;
    printf("%p\n", p1);

    printf("%p\n", p2);

    printf("%d\n", &num);

}