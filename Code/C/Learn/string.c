#include<stdio.h>

int main(){
    char* arr[5] = {"apple", "banana", "cherry", "date", "elderberry"};
    char** p = arr;

    for(int i = 0; i < 5; i++){
        int j = 0;
        while(*(*(p + i) + j) != '\0'){
            printf("%c", *(*(p + i) + j));
            j++;
        }
        printf("\n");
    }
}