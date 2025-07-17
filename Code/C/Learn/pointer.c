#include <stdio.h>

void swap(int *a, int *b);
void max_and_min(int arr[], int len, int *max, int *min);
int divide(int a, int b, double* result);

int main() {
    int a = 10;
    int b = 20;
    int arr[] = {1, 2, 3, 4, 5};
    int len = sizeof(arr) / sizeof(arr[0]);

    printf("a: %d, b: %d\n", a, b);

    swap(&a, &b);
    printf("a: %d, b: %d\n", a, b);

    printf("--------------------------------\n");

    int max, min;
    max_and_min(arr, len, &max, &min);
    printf("max: %d, min: %d\n", max, min);

    printf("--------------------------------\n");

    double res = 0;
    int stat = divide(8, 3, &res);
    if(!stat) {
        printf("result: %f\n", res);
    } else {
        printf("error: division by zero\n");
    }

    printf("--------------------------------\n");

    int num1 = 8;
    int* p = &num1;
    
    printf("%d\n", sizeof(num1));
    printf("%p\n", p);
    printf("%p\n", p + 1);
    printf("%p\n", p + 2);

    printf("--------------------------------\n");

    int arr1[] = {1, 114, 3, 4, 5};
    int* p1 = &arr1[0];
    int* p2 = &arr1[4];
    printf("%d\n", *p1);
    printf("%d\n", *(p1 + 1));
    printf("%d\n", p2 - p1);
    return 0;
}

void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

void max_and_min(int arr[], int len, int *max, int *min) {
    *max = arr[0];
    *min = arr[0];
    for(int i = 0; i < len; i++) {
        if(arr[i] > *max) {
            *max = arr[i];
        }
        if(arr[i] < *min) {
            *min = arr[i];
        }
    }
}

int divide(int a, int b, double* result) {
    if(b == 0) {
        return -1;
    }
    
    *result = (double)a / b;
    return 0;
}