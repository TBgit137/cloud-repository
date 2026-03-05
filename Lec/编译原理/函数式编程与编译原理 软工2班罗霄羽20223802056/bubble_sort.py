def bubble_sort(arr):
    """
    对列表 arr 进行原地冒泡排序（从小到大）
    """
    n = len(arr)
    for i in range(n - 1):
        swapped = False
        # 内层循环：每一趟把当前未排序部分的最大值“冒泡”到末尾
        for j in range(0, n - 1 - i):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        # 如果本趟无交换，说明已整体有序，提前结束
        if not swapped:
            break


if __name__ == "__main__":
    raw = input("请输入要排序的整数列表（用空格分隔）：")
    arr = [int(x) for x in raw.split()] if raw.strip() else []
    print("原列表：", arr)

    bubble_sort(arr)

    print("排序后：", arr)
