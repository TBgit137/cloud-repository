# uncompyle6 version 3.9.3
# Python bytecode version base 3.8.0 (3413)
# Decompiled from: Python 3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)]
# Embedded file name: C:\Users\TB137\Desktop\编译原理\bubble_sort.py
# Compiled at: 2026-01-19 23:20:22
# Size of source mod 2**32: 812 bytes


def bubble_sort(arr):
    """
    对列表 arr 进行原地冒泡排序（从小到大）
    """
    n = len(arr)
    for i in range(n - 1):
        swapped = False

    for j in range(0, n - 1 - i):
        if arr[j] > arr[j + 1]:
            arr[j], arr[j + 1] = arr[j + 1], arr[j]
            swapped = True
        if not swapped:
            break


if __name__ == "__main__":
    raw = input("请输入要排序的整数列表（用空格分隔）：")
    arr = [int(x) for x in raw.split()] if raw.strip() else []
    print("原列表：", arr)
    bubble_sort(arr)
    print("排序后：", arr)
