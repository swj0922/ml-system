import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def split_data(input_file, train_output_file, test_output_file, test_size=0.2, random_state=42):
    """
    将数据集分割为训练集和测试集
    
    参数:
    - input_file: 输入数据文件路径
    - train_output_file: 训练集输出文件路径
    - test_output_file: 测试集输出文件路径
    - test_size: 测试集比例，默认为0.2
    - random_state: 随机种子，默认为42
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(train_output_file), exist_ok=True)
    os.makedirs(os.path.dirname(test_output_file), exist_ok=True)
    
    # 读取数据
    print(f"正在读取数据文件: {input_file}")
    data = pd.read_csv(input_file)
    
    # 显示数据基本信息
    print(f"数据集形状: {data.shape}")
    print(f"数据集列数: {len(data.columns)}")
    print(f"目标列 'Bankrupt' 的分布:\n{data['Bankrupt'].value_counts()}")
    
    # 分割数据
    print(f"按 {test_size*100}% 的比例分割数据...")
    train_data, test_data = train_test_split(
        data, 
        test_size=test_size, 
        random_state=random_state,
        stratify=data['Bankrupt']  # 保持目标变量的分布比例
    )
    
    # 保存分割后的数据
    print(f"保存训练集到: {train_output_file}")
    train_data.to_csv(train_output_file, index=False)
    
    print(f"保存测试集到: {test_output_file}")
    test_data.to_csv(test_output_file, index=False)
    
    # 显示分割结果统计信息
    print("\n分割结果统计:")
    print(f"训练集形状: {train_data.shape}")
    print(f"测试集形状: {test_data.shape}")
    print(f"训练集中 'Bankrupt' 的分布:\n{train_data['Bankrupt'].value_counts()}")
    print(f"测试集中 'Bankrupt' 的分布:\n{test_data['Bankrupt'].value_counts()}")
    
    # 计算并显示分割比例
    train_ratio = len(train_data) / len(data)
    test_ratio = len(test_data) / len(data)
    print(f"\n实际分割比例 - 训练集: {train_ratio:.2%}, 测试集: {test_ratio:.2%}")
    
    print("\n数据分割完成!")

if __name__ == "__main__":
    # 设置文件路径
    input_csv = "data/data.csv"
    train_csv = "data/train_data.csv"
    test_csv = "data/test_data.csv"
    
    # 执行数据分割
    split_data(input_csv, train_csv, test_csv)