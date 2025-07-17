#include <stdio.h>

int main() {

}

void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

// senario: Small datasets/Nearly sorted
void bubble_sort(int arr[], int len) {
    // time complexity: O(n^2)
    // space complexity: O(1)
    // stable: yes
    for(int i = 0; i < len; i++) {
        int flag = 0;
        for(int j = 0; j < len - 1; j++) {
            if(arr[j] > arr[j + 1]) {
                swap(&arr[j], &arr[j + 1]);
                flag = 1;
            }
        }

        if(flag == 0) {
            break;
        }
    }
}

// senario: Small datasets + high swap cost
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
            swap(&arr[i], &arr[min]);
        }
    }
}

// senario: Small/partially sorted data, Real-time streaming data
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

// senario: Medium-sized datasets, Memory-constrained + better than O(n²), Legacy systems
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

void heapfiy(int arr[], int len, int i) {
    int largest = i;
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    // check if the left child or right child is larger
    if(left < len && arr[left] > arr[largest]) {
        largest = left;
    }
    if(right < len && arr[right] > arr[largest]) {
        largest = right;
    }

    // if the largest is not the root, swap the root with the largest
    if(largest != i) {
        swap(&arr[i], &arr[largest]);
        heapfiy(arr, len, largest);
    }
}

void build_max_heap(int arr[], int len) {
    for(int i = len / 2 - 1; i >= 0; i--) {
        heapfiy(arr, len, i);
    }
}

// senario: Guaranteed worst-case performance, Large memory-sensitive datasets, Real-time systems
void heap_sort(int arr[], int len) {
    // time complexity: O(nlogn)
    // space complexity: O(1)
    // stable: no
    build_max_heap(arr, len);

    for(int i = len - 1; i >= 0; i--) {
        swap(&arr[0], &arr[i]);
        heapfiy(arr, i, 0);
    }
}

int partition(int arr[], int left, int right) {
    int pivot = arr[right];
    int i = left - 1;

    for(int j = left; j < right; j++) {
        if(arr[j] <= pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[right]);
    return i + 1;
}

// senario: Large random datasets, Standard library implementations, Optimized to avoid O(n²)
void quick_sort(int arr[], int left, int right) {
    // time complexity: O(nlogn)
    // space complexity: O(logn)
    // stable: no
    if (left < right) {
        int pi = partition(arr, left, right);
        
        quick_sort(arr, left, pi - 1);

        quick_sort(arr, pi + 1, right);
    }
}

void merge(int arr[], int left, int mid, int right) {
    int i, j, k;
    int n1 = mid - left + 1;
    int n2 = right - mid;

    int *L = (int *)malloc(n1 * sizeof(int));
    int *R = (int *)malloc(n2 * sizeof(int));

    for(i = 0; i < n1; i++) {
        L[i] = arr[left + i];
    }
    for(j = 0; j < n2; j++) {
        R[j] = arr[mid + 1 + j];
    }

    i = 0;
    j = 0;
    k = left;

    while(i < n1 && j < n2) {
        if(L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while(i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while(j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }

    free(L);
    free(R);
}

// senario: Stable sorting required, External/large-data sorting, Linked lists/distributed systems
void merge_sort(int arr[], int left, int right) {
    // time complexity: O(nlogn)
    // space complexity: O(n)
    // stable: yes
    if(left < right) {
        int mid = left + (right - left) / 2;
        merge_sort(arr, left, mid);
        merge_sort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}