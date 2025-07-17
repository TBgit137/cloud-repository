import os
import glob

# 获取当前目录下所有.exe文件
exe_files = glob.glob("*.exe")

for file in exe_files:
    try:
        os.remove(file)
        print(f"已删除: {file}")
    except Exception as e:
        print(f"删除{file}时出错: {e}")