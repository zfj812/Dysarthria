import os
from collections import defaultdict

def clean_files(directory):
    """
    清洗目录下的文件：
    - 每个任务只保留一个文件，优先保留编号为 `_1` 的文件；
    - 如果任务文件没有编号（无后缀），直接保留。
    
    :param directory: 要清洗的目录路径
    """
    # 创建一个字典来记录每个任务的文件
    task_files = defaultdict(list)
    
    # 遍历文件夹下的所有文件
    for filename in os.listdir(directory):
        # 判断是否有后缀
        parts = filename.rsplit("_", maxsplit=1)
        if len(parts) == 2 :  # 带后缀的文件
            task_name = parts[0]  # 提取任务名（不包括编号部分）
        else:  # 无后缀的文件
            task_name = filename.rsplit(".", maxsplit=1)[0]  # 提取文件名去掉扩展名
        
        task_files[task_name].append(filename)
    
    # 保留每个任务的一个文件
    files_to_keep = set()
    files_to_remove = set()
    for task_name, files in task_files.items():
        if len(files) == 1:  # 如果只有一个文件，直接保留
            files_to_keep.add(files[0])
        else:  # 如果有多个文件
            files.sort()  # 按字典序排序
            # 优先保留编号为 _1 的文件，若无则保留排序中的第二个
            keep_file = next((f for f in files if "_1" in f), files[min(1, len(files) - 1)])
            files_to_keep.add(keep_file)
            files_to_remove.update(f for f in files if f != keep_file)

    # 删除多余文件
    for filename in files_to_remove:
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"已删除文件: {filename}")

    print("保留的文件:")
    for filename in files_to_keep:
        print(filename)

# 使用示例
directory_path = r"E:\lyh\data\beijingdata\alldata\video_preprocessed_256\lip\normalize"
clean_files(directory_path)


