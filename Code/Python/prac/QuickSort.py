def quicksort(array):
    if len(array) < 2:
        return array
    pivot = array[len(array) // 2]
    left = [x for x in array if x < pivot]
    right = [x for x in array if x > pivot]
    middle = [x for x in array if x == pivot]
    return quicksort(left) + middle + quicksort(right)