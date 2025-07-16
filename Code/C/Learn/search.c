#include <stdio.h>

int bi_search(int arr[], int target, int len);

int main() {
    int arr[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    int target = 5;

    int len = sizeof(arr) / sizeof(arr[0]);

    int result = bi_search(arr, target, len);

    printf("%d\n", result);

    return 0;

}

int bi_search(int arr[], int target, int len) {
    int min = 0, max = len - 1;

    while(min <= max) {
        int mid = (min + max) / 2;

        if(arr[mid] == target) {
            return mid;
        }
        else if(arr[mid] > target) {
            max = mid - 1;
        }
        else if(arr[mid] < target) {
            min = mid + 1;
        }
    }

    return -1;
}