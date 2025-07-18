#include<stdio.h>

typedef struct {
    char name[20];
    int age;
    int score;
    char gender;
} st;

void printStudent(st stu);
void editStudent(st* p_stu);

int main(){
    st larry = {"larry", 20, 90, 'M'};
    st steven = {"steven", 21, 85, 'M'};
    st james = {"james", 22, 88, 'M'};

    st arr[3] = {larry, steven, james};

    for(int i = 0; i < 3; i++){
        printf("name: %s, age: %d, score: %d, gender: %c\n", arr[i].name, arr[i].age, arr[i].score, arr[i].gender);
    }

    printf("--------------------------------\n");

    printStudent(larry);
    editStudent(&larry);
    printStudent(larry);

    return 0;
}

void printStudent(st stu){
    printf("%s, %d\n", stu.name, stu.age);
}

void editStudent(st* p_stu){
    p_stu -> age = 100;
}