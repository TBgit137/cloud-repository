#include <stdio.h>

int main(void){
    int num;

    printf("Enter number: ");
    scanf("%d", &num);

    int sum = 0;
    for(int i = 0; i <= num; i+=2){
        sum += i;
    }

    printf("%d", sum);

    return 0;
}