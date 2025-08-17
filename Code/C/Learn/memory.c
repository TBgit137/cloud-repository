#include <stdio.h>
#include <stdlib.h>

int main(){

    int *p =  malloc(10 * sizeof(int)); 
    // in bytes(memory space for 40 bytes)
    // return a pointer in void
    // only return a pointer to the first byte of the memory block, better add an attribute to show the size of the block
    // remember to free the memory after use
    // too much application for memory will trigger virtual memory
    // after free, data in the remaining memory is called dirty data

    printf("p: %p\n", p);

    for(int i = 0; i < 10; i++){
        p[i] = (i + 1) * 10;
    }

    for(int i = 0; i < 100; i++){
        printf("%d ", *(p + i));
    }

    printf("--------------------------------\n");

    int *p2 = realloc(p, 20 * sizeof(int));

    for(int i = 0; i < 20; i++){
        printf("%d ", *(p2 + i));
    }

    free(p2);
    // after realloc, do not need to free the original pointer

    return 0;

}