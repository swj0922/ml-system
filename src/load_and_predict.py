"""
加载和使用训练好的XGBoost模型示例
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

def load_model(model_path):
    """
    加载保存的模型
    
    参数:
    model_path: 模型文件路径
    
    返回:
    加载的模型
    """
    model = joblib.load(model_path)
    print(f"模型已从 {model_path} 加载")
    return model

def predict_with_model(model, data_path):
    """
    使用模型进行预测
    
    参数:
    model: 加载的模型
    data_path: 测试数据文件路径
    
    返回:
    预测结果和概率
    """
    # 加载测试数据
    data = pd.read_csv(data_path)
    
    # 分离特征和目标变量
    X_test = data.iloc[:, :-1]
    y_test = data.iloc[:, -1]
    
    print(f"测试数据加载完成，共 {len(data)} 条记录")
    
    # 进行预测
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
    
    # 返回预测结果和概率
    return y_pred, y_pred_proba, metrics

def main():
    # 模型文件路径
    model_path = 'models/xgboost_binned_model.joblib'
    
    # 测试数据文件路径
    test_data_path = 'data/binned_test_data.csv'
    
    # 加载模型
    model = load_model(model_path)
    
    # 使用模型进行预测
    y_pred, y_pred_proba, metrics = predict_with_model(model, test_data_path)
    
    # 显示一些预测结果示例
    print("\n预测结果示例 (前10条):")
    for i in range(10):
        print(f"样本 {i+1}: 预测类别={y_pred[i]}, 预测概率={y_pred_proba[i]:.4f}")

if __name__ == "__main__":
    main()