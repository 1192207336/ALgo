import os
import argparse


def rename_files_to_json(folder_path, dry_run=False):
    """
    将指定文件夹下的所有文件后缀改为.json

    参数:
    folder_path (str): 目标文件夹路径
    dry_run (bool): 是否仅预览而不实际重命名 (默认False)
    """
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在")
        return

    if not os.path.isdir(folder_path):
        print(f"错误: '{folder_path}' 不是文件夹")
        return

    print(f"处理文件夹: {folder_path}")
    if dry_run:
        print(">>> 预览模式 (不会实际修改文件) <<<")

    # 计数器
    renamed_count = 0

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 只处理文件（跳过目录）
        if os.path.isfile(file_path):
            # 分离文件名和扩展名
            name, ext = os.path.splitext(filename)

            # 构建新的文件名（保留原文件名，只改后缀）
            new_filename = name + ".json"
            new_file_path = os.path.join(folder_path, new_filename)

            # 检查是否已存在同名文件
            if os.path.exists(new_file_path):
                print(f"警告: 无法重命名 '{filename}' -> '{new_filename}' (目标文件已存在)")
                continue

            # 执行重命名或预览
            if dry_run:
                print(f"[预览] 重命名: {filename} -> {new_filename}")
            else:
                try:
                    os.rename(file_path, new_file_path)
                    print(f"已重命名: {filename} -> {new_filename}")
                    renamed_count += 1
                except Exception as e:
                    print(f"错误: 无法重命名 '{filename}' -> '{new_filename}': {str(e)}")

    # 输出总结
    print("\n操作完成:")
    if dry_run:
        print(f"预览了 {renamed_count} 个可重命名的文件")
    else:
        print(f"成功重命名了 {renamed_count} 个文件")


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description='将指定文件夹下的所有文件后缀改为.json',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'folder',
        type=str,
        help='目标文件夹路径'
    )
    parser.add_argument(
        '-d', '--dry-run',
        action='store_true',
        help='预览模式，仅显示将要执行的操作而不实际修改文件'
    )

    # 解析命令行参数
    args = parser.parse_args()

    # 执行重命名操作
    rename_files_to_json(args.folder, args.dry_run)