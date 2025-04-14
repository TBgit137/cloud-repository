def mergesort(array):
    if len(array) < 2:
        return array
    mid = len(array) // 2
    left = mergesort(array[:mid])
    right = mergesort(array[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    while len(left) > 0 and len(right) > 0:
        if left[0] <= right[0]:
            result.append(left.pop(0))
        else:
            result.append(right.pop(0))
    result.extend(left)
    result.extend(right)
    return result