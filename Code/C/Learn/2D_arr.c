#include <stdio.h>

int main() {
    int arr[3][3] = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            printf("%d", arr[i][j]);
        }
        printf("\n");
    }

    printf("--------------------------------\n");

    int arr_1d_1[] = {1, 2, 3};
    int arr_1d_2[] = {4, 5, 6, 7, 8};
    int arr_1d_3[] = {9, 10, 11, 12, 13};

    int* arr_2d[] = {arr_1d_1, arr_1d_2, arr_1d_3};

    int arr_len[] = {sizeof(arr_1d_1) / sizeof(arr_1d_1[0]), 
                   sizeof(arr_1d_2) / sizeof(arr_1d_2[0]), 
                   sizeof(arr_1d_3) / sizeof(arr_1d_3[0])};

    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < arr_len[i]; j++) {
            printf("%d", arr_2d[i][j]);
        }
        printf("\n");
    }

    printf("--------------------------------\n");

    int(*p_2d)[3] = arr;

    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            printf("%d", *(*(p_2d + i) + j));
        }
        printf("\n");
    }

    printf("--------------------------------\n");

    int arr1[5] = {1, 2, 3, 4, 5};
    int arr2[5] = {6, 7, 8, 9, 10};
    int* arr_big[] = {arr1, arr2};
    int **p_big = arr_big;

    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 5; j++) {
            printf("%d", *(*(p_big + i) + j));
        }
        printf("\n");
    }

    return 0;
}