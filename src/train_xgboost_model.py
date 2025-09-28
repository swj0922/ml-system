"""
基于binned_train_data.csv文件构建XGBoost模型并保存
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
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

def evaluate_model(model, X_test, y_test):
    """
    评估模型性能
    
    参数:
    model: 训练好的模型
    X_test: 测试特征
    y_test: 测试目标
    
    返回:
    metrics: 评估指标字典
    """
    # 预测
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 计算评估指标
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'pr_auc': average_precision_score(y_test, y_pred_proba)
    }
    
    # 打印评估结果
    print("\n模型评估结果:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # 打印分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    # 绘制混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return metrics

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

def plot_feature_importance(model, top_n=20):
    """
    绘制特征重要性
    
    参数:
    model: 训练好的模型
    top_n: 显示前N个重要特征
    """
    # 获取特征重要性
    importance = model.feature_importances_
    feature_names = model.feature_names_in_
    
    # 创建DataFrame并排序
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # 取前N个特征
    top_features = feature_importance.head(top_n)
    
    # 绘制特征重要性图
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=top_features)
    plt.title(f'前 {top_n} 个重要特征')
    plt.xlabel('重要性')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    return feature_importance

def main():
    # 数据文件路径
    data_path = 'data/binned_train_data.csv'
    
    # 加载数据
    X, y = load_data(data_path)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 进一步划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
    )
    
    print(f"训练集大小: {X_train.shape}")
    print(f"验证集大小: {X_val.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 训练模型
    print("\n开始训练XGBoost模型...")
    model = train_xgboost_model(X_train, y_train, X_val, y_val)
    
    # 评估模型
    metrics = evaluate_model(model, X_test, y_test)
    
    # 绘制特征重要性
    feature_importance = plot_feature_importance(model)
    
    # 保存模型
    model_path = 'models/xgboost_binned_model.joblib'
    save_model(model, model_path)
    
    # 保存特征重要性
    feature_importance.to_csv('feature_importance.csv', index=False)
    print("特征重要性已保存到: feature_importance.csv")
    
    print("\n模型训练和评估完成!")

if __name__ == "__main__":
    main()