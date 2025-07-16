#include <stdio.h>

int main() {

}

void bubble_sort(int arr[], int len) {
    // time complexity: O(n^2)
    // space complexity: O(1)
    // stable: yes
    for(int i = 0; i < len; i++) {
        int flag = 0;
        for(int j = 0; j < len - 1; j++) {
            if(arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
                flag = 1;
            }
        }

        if(flag == 0) {
            break;
        }
    }
}

void select_sort(int arr[], int len) {
    // time complexity: O(n^2)
    // space complexity: O(1)
    // stable: no
    for(int i = 0; i < len - 1; i++) {
        int min = i;
        for(int j = i + 1; j < len; j++) {
            if(arr[j] < arr[min]) {
                min = j;
            }
        }

        if(min != i) {
            int temp = arr[i];
            arr[i] = arr[min];
            arr[min] = temp;
        }
    }
}

void insert_sort(int arr[], int len) {
    // time complexity: O(n^2)
    // space complexity: O(1)
    // stable: yes
    for(int i = 1; i < len; i++) {
        int temp = arr[i];
        int j = i - 1;
        while(j >= 0 && arr[j] > temp) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = temp;
    }
}

void hill_sort(int arr[], int len) {
    // time complexity: O(n^1.5)
    // space complexity: O(1)
    // stable: no
    int gap = len / 2;
    while(gap > 0) {
        for(int i = gap; i < len; i++) {
            int temp = arr[i];
            int j = i - gap;
            while(j >= 0 && arr[j] > temp) {
                arr[j + gap] = arr[j];
                j -= gap;
            }
            arr[j + gap] = temp;
        }
        gap /= 2;
    }
}