import os
import json
import pickle
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path


def merge_json_to_pkl(target_folder):
    """
    处理目标文件夹中的JSON文件：
    1. 将所有JSON文件内容合并到新的emb_81_32.json文件
    2. 将合并后的JSON文件转换为emb_81_32.pkl文件
    """
    target_path = Path(target_folder)

    # 步骤1: 检查目标文件夹
    if not target_path.exists():
        print(f"错误: 目标文件夹 '{target_folder}' 不存在")
        return
    if not target_path.is_dir():
        print(f"错误: '{target_folder}' 不是文件夹")
        return

    print(f"开始处理文件夹: {target_path}")

    # 步骤2: 收集所有JSON文件
    json_files = list(target_path.glob("*.json"))
    print(f"找到 {len(json_files)} 个JSON文件")

    if not json_files:
        print("警告: 没有找到JSON文件，操作终止")
        return

    # 步骤3: 创建合并的JSON文件
    merged_json_path = target_path / "emb_81_32.json"
    total_lines = 0

    # 先计算总行数用于进度条
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            total_lines += sum(1 for _ in f)

    print(f"开始合并 {len(json_files)} 个文件到 {merged_json_path}")

    # 实际合并文件
    with open(merged_json_path, 'w', encoding='utf-8') as outfile:
        for json_file in tqdm(json_files, desc="合并JSON文件"):
            with open(json_file, 'r', encoding='utf-8') as infile:
                for line in infile:
                    outfile.write(line)

    print(f"成功创建合并文件: {merged_json_path}")
    print(f"总行数: {total_lines}")

    # 步骤4: 将JSON转换为PKL
    pkl_path = target_path / "emb_81_32.pkl"
    emb_dict = {}
    processed_count = 0

    print(f"开始转换 {merged_json_path} 为PKL格式")

    with open(merged_json_path, 'r', encoding='utf-8') as json_file:
        for line in tqdm(json_file, total=total_lines, desc="转换JSON到PKL"):
            try:
                data = json.loads(line.strip())
                creative_id = data.get('anonymous_cid')
                emb = data.get('emb')

                if creative_id and emb:
                    # 确保嵌入是numpy数组
                    if isinstance(emb, list):
                        emb = np.array(emb, dtype=np.float32)
                    emb_dict[creative_id] = emb
                    processed_count += 1
            except json.JSONDecodeError:
                print(f"警告: 跳过无效的JSON行: {line[:50]}...")
            except Exception as e:
                print(f"处理行时出错: {str(e)}")

    # 保存为PKL文件
    with open(pkl_path, 'wb') as pkl_file:
        pickle.dump(emb_dict, pkl_file)

    print(f"成功创建PKL文件: {pkl_path}")
    print(f"嵌入向量总数: {len(emb_dict)}")
    print(f"处理的行数: {processed_count}")
    print("操作完成!")


if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(
        description='将目标文件夹中的所有JSON文件合并为emb_81_32.json并转换为emb_81_32.pkl',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'folder',
        type=str,
        help='目标文件夹路径'
    )

    # 解析参数
    args = parser.parse_args()

    # 执行处理
    merge_json_to_pkl(args.folder)