#include <stdio.h>
#include <unistd.h>

int* dangle_sim ();

int main() {
    int num1 = 11;
    int* p1 = &num1;
    printf("%d\n", *p1);
    printf("%p\n", p1);

    int* wild_p = p1 + 1;
    printf("%p\n", wild_p);
    printf("%d\n", *wild_p);

    int* dangle_p = dangle_sim();

    sleep(1);

    printf("%d\n", *dangle_p);
    printf("%p\n", dangle_p);

    return 0;
}

int* dangle_sim () {
    int num = 10;
    int* ptr = &num;
    return ptr;
}