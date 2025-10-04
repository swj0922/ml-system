"""
基于binned_train_data.csv文件构建多种机器学习模型并保存
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def load_data(file_path):
    """
    加载数据
    
    参数:
    file_path: 数据文件路径
    
    返回:
    X: 特征数据
    y: 目标变量
    """
    # 读取CSV文件
    data = pd.read_csv(file_path)
    
    # 分离特征和目标变量
    # 假设最后一列是目标变量'Bankrupt'
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    
    print(f"数据加载完成，共 {len(data)} 条记录，{X.shape[1]} 个特征")
    print(f"目标变量分布: {y.value_counts().to_dict()}")
    
    return X, y

def train_xgboost_model(X_train, y_train, X_val, y_val):
    """
    训练XGBoost模型
    
    参数:
    X_train: 训练特征
    y_train: 训练目标
    X_val: 验证特征
    y_val: 验证目标
    
    返回:
    model: 训练好的模型
    eval_results: 评估结果
    """
    # 创建XGBoost模型
    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=50
    )
    
    # 训练模型
    eval_set = [(X_train, y_train), (X_val, y_val)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=False
    )
    
    return model

def train_logistic_regression_model(X_train, y_train):
    """
    训练逻辑回归模型
    
    参数:
    X_train: 训练特征
    y_train: 训练目标
    
    返回:
    model: 训练好的模型
    """
    # 创建逻辑回归模型
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        solver='liblinear',
        class_weight='balanced'
    )
    
    # 训练模型
    model.fit(X_train, y_train)
    
    return model

def train_random_forest_model(X_train, y_train):
    """
    训练随机森林模型
    
    参数:
    X_train: 训练特征
    y_train: 训练目标
    
    返回:
    model: 训练好的模型
    """
    # 创建随机森林模型
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    
    # 训练模型
    model.fit(X_train, y_train)
    
    return model

def save_model(model, file_path):
    """
    保存模型到文件
    
    参数:
    model: 要保存的模型
    file_path: 保存路径
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # 保存模型
    joblib.dump(model, file_path)
    print(f"模型已保存到: {file_path}")


def main():
    # 数据文件路径
    data_path = 'data/binned_train_data.csv'
    
    # 加载数据
    X, y = load_data(data_path)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 1. 训练逻辑回归模型
    print("\n开始训练逻辑回归模型...")
    lr_model = train_logistic_regression_model(X_train, y_train)
    save_model(lr_model, 'models/logistic_model.joblib')
    
    # 2. 训练随机森林模型
    print("\n开始训练随机森林模型...")
    rf_model = train_random_forest_model(X_train, y_train)
    save_model(rf_model, 'models/rf_model.joblib')

    # 进一步划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
    )    
    
    # 3. 训练XGBoost模型
    print("\n开始训练XGBoost模型...")
    xgb_model = train_xgboost_model(X_train, y_train, X_val, y_val)
    save_model(xgb_model, 'models/xgb_model.joblib')
    
    print("\n所有模型训练和评估完成!")

if __name__ == "__main__":
    main()