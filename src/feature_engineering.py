import pandas as pd
import numpy as np
import os

def load_data(file_path):
    """加载数据集"""
    return pd.read_csv(file_path)

def create_derived_features(df):
    """创建衍生特征"""
    # 获取特征列（排除标签列'Bankrupt'）
    feature_columns = [col for col in df.columns if col != 'Bankrupt']
    
    # 创建一个新的DataFrame来存储原始特征和衍生特征
    df_derived = df.copy()
    
    # 1. 特征间的乘法运算
    for i in range(len(feature_columns)):
        for j in range(i+1, len(feature_columns)):
            feat1 = feature_columns[i]
            feat2 = feature_columns[j]
            # 创建新的特征列名
            new_feat_name = f"{feat1}_x_{feat2}"
            # 计算乘积
            df_derived[new_feat_name] = df[feat1] * df[feat2]
    
    # 2. 特征的平方运算
    for feat in feature_columns:
        new_feat_name = f"{feat}_squared"
        df_derived[new_feat_name] = df[feat] ** 2
    
    # 3. 特征间的除法运算（避免除以零）
    for i in range(len(feature_columns)):
        for j in range(len(feature_columns)):
            if i != j:
                feat1 = feature_columns[i]
                feat2 = feature_columns[j]
                # 创建新的特征列名
                new_feat_name = f"{feat1}_div_{feat2}"
                # 计算除法，添加一个小常数避免除以零
                df_derived[new_feat_name] = df[feat1] / (df[feat2] + 1e-8)
    
    # 4. 特征的对数变换（确保所有值为正）
    for feat in feature_columns:
        new_feat_name = f"{feat}_log"
        # 添加一个小常数确保所有值为正
        df_derived[new_feat_name] = np.log1p(df[feat])
    
    # 5. 特征的平方根变换（确保所有值为非负）
    for feat in feature_columns:
        new_feat_name = f"{feat}_sqrt"
        # 取绝对值确保所有值为非负
        df_derived[new_feat_name] = np.sqrt(np.abs(df[feat]))
    
    return df_derived

def save_derived_data(df, output_path):
    """保存衍生特征数据"""
    df.to_csv(output_path, index=False)
    print(f"衍生特征数据已保存到: {output_path}")

def main():
    # 定义文件路径
    train_path = "data/selected_train_data.csv"
    test_path = "data/selected_test_data.csv"
    derived_train_path = "data/derived_train_data.csv"
    derived_test_path = "data/derived_test_data.csv"
    
    # 加载数据
    print("加载训练集...")
    train_df = load_data(train_path)
    print("加载测试集...")
    test_df = load_data(test_path)
    
    # 创建衍生特征
    print("为训练集创建衍生特征...")
    train_derived = create_derived_features(train_df)
    print("为测试集创建衍生特征...")
    test_derived = create_derived_features(test_df)
    
    # 保存衍生特征数据
    print("保存训练集衍生特征...")
    save_derived_data(train_derived, derived_train_path)
    print("保存测试集衍生特征...")
    save_derived_data(test_derived, derived_test_path)
    
    # 输出数据集形状信息
    print(f"原始训练集形状: {train_df.shape}")
    print(f"衍生训练集形状: {train_derived.shape}")
    print(f"原始测试集形状: {test_df.shape}")
    print(f"衍生测试集形状: {test_derived.shape}")

if __name__ == "__main__":
    main()