#include <stdio.h>

void swap(void* p1, void* p2, int len);

int main() {
    long num1 = 10;
    long num2 = 20;

    printf("num1: %ld, num2: %ld\n", num1, num2);

    swap(&num1, &num2, sizeof(long));

    printf("num1: %ld, num2: %ld\n", num1, num2);
}

void swap(void* p1, void* p2, int len) {
    char* pc1 = p1;
    char* pc2 = p2;
    char temp = 0;

    for(int i = 0; i < len; i++) {
        temp = pc1[i];
        pc1[i] = pc2[i];
        pc2[i] = temp;
    }
}