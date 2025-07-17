#include <stdio.h>

int main() {
    int arr[3] = {1, 2, 3};
    int len = sizeof(arr) / sizeof(arr[0]);
    int* p_arr = arr;
    int* p_arr_head = &arr[0];
    int* p_arr_second = &arr[1];

    printf("%p\n", p_arr);
    printf("%p\n", p_arr_head);
    printf("%p\n", p_arr_second);

    printf("--------------------------------\n");

    p_arr += 1;
    
    printf("%p\n", p_arr);
    printf("%p\n", p_arr_head);
    printf("%p\n", p_arr_second);

    return 0;
}