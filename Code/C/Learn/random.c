#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    srand(time(NULL));

    int count = 0;

    while(count < 10) {
        int random_num = rand() % 100 + 1;
        printf("%d\n", random_num);
        count++;
    }

    return 0;
}