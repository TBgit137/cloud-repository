#include <stdio.h>

int main(void){
    int height;

    printf("Enter height(m): ");
    scanf("%d", &height);

    float paper_height = 0.0001;
    int count = 0;
    while(paper_height < height){
        paper_height *= 2;
        count++;
    }

    printf("fold times: %d", count);
}