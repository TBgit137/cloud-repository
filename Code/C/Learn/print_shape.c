#include <stdio.h>

void print_rectangle(int length, int width){
    for(int i = 0; i < width; i++){
        for(int j = 0; j < length; j++){
            printf("*");
        }
        printf("\n");
    }
}

void print_triangle(int length, int width) {
    for (int i = 0; i < length; i++) {
        int stars = (i + 1) * width / length;
        if (stars == 0) stars = 1;
        for (int j = 0; j < stars; j++) {
            printf("*");
        }
        printf("\n");
    }
}

int main(void){
    char shape;
    int length, width;

    printf("Enter shape (r for rectangle, t for triangle): ");
    scanf("%c", &shape);

    printf("Enter length: ");
    scanf("%d", &length);

    printf("Enter width: ");
    scanf("%d", &width);

    if(shape == 'r'){
        print_rectangle(length, width);
    }
    else if(shape == 't'){
        print_triangle(length, width);
    }

    return 0;
}