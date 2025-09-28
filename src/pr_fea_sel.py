"""
增强版XGBoost特征选择与可视化（使用PR曲线下面积作为评估指标）
基于data.csv文件训练XGBoost模型，提取特征重要性，并通过前向特征选择筛选最优特征子集，
使用PR曲线下面积作为评估指标，并调整特征筛选过程的终止条件。
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')

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
    feature_names: 特征名称列表
    """
    # 读取数据
    data = pd.read_csv(file_path)
    
    # 分离特征和目标变量
    y = data['Bankrupt'].values
    X = data.drop('Bankrupt', axis=1).values
    feature_names = data.drop('Bankrupt', axis=1).columns.tolist()
    
    return X, y, feature_names

def train_xgboost_model(X, y):
    """
    训练XGBoost模型并提取特征重要性
    
    参数:
    X: 特征数据
    y: 目标变量
    
    返回:
    model: 训练好的XGBoost模型
    feature_importance: 特征重要性字典
    """
    # 创建XGBoost模型
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    # 训练模型
    model.fit(X, y)
    
    # 获取特征重要性
    feature_importance = model.feature_importances_
    
    return model, feature_importance

def forward_feature_selection(X, y, feature_names, feature_importance, cv=5):
    """
    前向特征选择（使用PR曲线下面积作为评估指标）
    
    参数:
    X: 特征数据
    y: 目标变量
    feature_names: 特征名称列表
    feature_importance: 特征重要性
    cv: 交叉验证折数
    
    返回:
    selected_features: 选中的特征索引列表
    pr_scores: 每次迭代的PR曲线下面积分数列表
    feature_history: 每次迭代添加的特征历史
    """
    # 初始化
    selected_features = []  # 选中的特征索引
    remaining_features = list(range(len(feature_names)))  # 剩余特征索引
    pr_scores = []  # 每次迭代的PR曲线下面积分数
    feature_history = []  # 每次迭代添加的特征历史
    
    # 创建交叉验证对象
    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    # 记录最佳PR分数和对应的特征集
    best_pr_score = -np.inf
    best_feature_set = []
    
    # 记录连续未提升的轮数
    no_improvement_count = 0
    
    # 不断添加特征，直到没有特征可以添加或者满足终止条件
    while len(remaining_features) > 0:
        best_pr = -np.inf
        best_feature = None
        
        # 尝试添加每个剩余特征
        for feature in remaining_features:
            # 临时特征集
            temp_features = selected_features + [feature]
            temp_X = X[:, temp_features]
            
            # 计算交叉验证PR曲线下面积
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            
            # 使用StratifiedKFold进行交叉验证
            skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
            pr_auc_scores = []
            
            for train_idx, val_idx in skf.split(temp_X, y):
                X_train, X_val = temp_X[train_idx], temp_X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # 训练模型
                model.fit(X_train, y_train)
                
                # 预测概率
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                
                # 计算PR AUC
                pr_auc = average_precision_score(y_val, y_pred_proba)
                pr_auc_scores.append(pr_auc)
            
            # 计算平均PR AUC
            mean_pr_auc = np.mean(pr_auc_scores)
            
            # 如果这个特征能带来最大的PR提升，则选择它
            if mean_pr_auc > best_pr:
                best_pr = mean_pr_auc
                best_feature = feature
        
        # 添加最佳特征
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
        pr_scores.append(best_pr)
        feature_history.append(feature_names[best_feature])
        
        print(f"添加特征: {feature_names[best_feature]}, PR曲线下面积: {best_pr:.4f}")
        
        # 检查PR分数是否有提升
        if best_pr > best_pr_score:
            best_pr_score = best_pr
            best_feature_set = selected_features.copy()
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        
        # 检查终止条件：连续3轮未提升
        if no_improvement_count >= 3:
            print(f"连续3轮PR曲线下面积未提升，终止特征选择")
            print(f"回溯至最佳特征集，PR曲线下面积: {best_pr_score:.4f}")
            break
    
    # 如果因为连续3轮未提升而终止，则回溯至最佳特征集
    if no_improvement_count >= 3:
        # 找到最佳特征集对应的PR分数历史
        best_pr_scores = pr_scores[:len(best_feature_set)]
        best_feature_history = feature_history[:len(best_feature_set)]
        return best_feature_set, best_pr_scores, best_feature_history
    
    return selected_features, pr_scores, feature_history

def calculate_feature_correlation(X, selected_features):
    """
    计算选中特征之间的相关性
    
    参数:
    X: 特征数据
    selected_features: 选中的特征索引列表
    
    返回:
    corr_matrix: 相关性矩阵
    """
    # 提取选中特征的数据
    selected_X = X[:, selected_features]
    
    # 计算相关性矩阵
    corr_matrix = np.corrcoef(selected_X, rowvar=False)
    
    return corr_matrix

def visualize_results(feature_names, feature_importance, selected_features, pr_scores, feature_history, X, data_type="train"):
    """
    可视化特征选择结果，输出两张独立图像
    
    参数:
    feature_names: 特征名称列表
    feature_importance: 特征重要性
    selected_features: 选中的特征索引列表
    pr_scores: 每次迭代的PR曲线下面积分数列表
    feature_history: 每次迭代添加的特征历史
    X: 特征数据
    data_type: 数据类型，用于区分训练集和测试集
    """
    # 确保results文件夹存在
    import os
    if not os.path.exists('d:/桌面/学习资料/项目/ml/results'):
        os.makedirs('d:/桌面/学习资料/项目/ml/results')
    
    # 第一张图：特征重要性柱状图与PR曲线下面积折线图整合
    plt.figure(figsize=(15, 8))
    
    # 获取选中特征的重要性
    selected_importance = [feature_importance[i] for i in selected_features]
    selected_names = [feature_names[i] for i in selected_features]
    
    # 归一化特征重要性
    normalized_importance = np.array(selected_importance) / np.sum(selected_importance)
    
    # 创建双Y轴
    ax1 = plt.gca()
    ax1_twin = ax1.twinx()
    
    # 绘制特征重要性柱状图
    colors = cm.viridis(np.linspace(0, 1, len(selected_features)))
    bars = ax1.bar(range(len(selected_features)), normalized_importance, color=colors, alpha=0.7)
    
    # 绘制PR曲线下面积折线图
    line = ax1_twin.plot(range(len(selected_features)), pr_scores, 'o-', color='red', linewidth=2, markersize=8, label='PR曲线下面积')
    
    # 设置X轴标签
    ax1.set_xticks(range(len(selected_features)))
    ax1.set_xticklabels(selected_names, rotation=45, ha='right')
    
    # 设置Y轴标签
    ax1.set_ylabel('归一化重要性', fontsize=12)
    ax1_twin.set_ylabel('PR曲线下面积', fontsize=12)
    
    # 添加标题
    ax1.set_title(f'特征重要性与PR曲线下面积 ({data_type}数据集)', fontsize=14, fontweight='bold')
    
    # 只添加PR曲线下面积折线图的图例
    ax1_twin.legend(loc='upper left')
    
    # 在柱状图上添加数值标签
    for bar, imp in zip(bars, normalized_importance):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{imp:.3f}',
                 ha='center', va='bottom')
    
    # 在折线图上添加PR分数标签
    for i, pr in enumerate(pr_scores):
        ax1_twin.text(i, pr + 0.001, f'{pr:.4f}', ha='center', va='bottom')
    
    # 添加网格线
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形到results文件夹
    plt.savefig(f'd:/桌面/学习资料/项目/ml/results/feature_importance_pr_scores_{data_type}.png', dpi=300, bbox_inches='tight')
    
    # 显示图形
    plt.show()
    
    # 第二张图：特征相关性热力图
    plt.figure(figsize=(10, 8))
    
    # 计算选中特征的相关性
    corr_matrix = calculate_feature_correlation(X, selected_features)
    
    # 创建热力图
    im = plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    
    # 添加颜色条
    cbar = plt.colorbar(im)
    cbar.set_label('相关系数', fontsize=12)
    
    # 设置刻度标签
    plt.xticks(range(len(selected_features)))
    plt.yticks(range(len(selected_features)))
    plt.xticks(range(len(selected_features)), [feature_names[i] for i in selected_features], rotation=45, ha='right')
    plt.yticks(range(len(selected_features)), [feature_names[i] for i in selected_features])
    
    # 添加标题
    plt.title(f'选中特征的相关性热力图 ({data_type}数据集)', fontsize=14, fontweight='bold')
    
    # 在热力图上添加相关系数值
    for i in range(len(selected_features)):
        for j in range(len(selected_features)):
            text = plt.text(j, i, f'{corr_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black" if abs(corr_matrix[i, j]) < 0.5 else "white")
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形到results文件夹
    plt.savefig(f'd:/桌面/学习资料/项目/ml/results/feature_correlation_heatmap_{data_type}.png', dpi=300, bbox_inches='tight')
    
    # 显示图形
    plt.show()

def print_feature_selection_summary(feature_names, feature_importance, selected_features, pr_scores, feature_history):
    """
    打印特征选择摘要
    
    参数:
    feature_names: 特征名称列表
    feature_importance: 特征重要性
    selected_features: 选中的特征索引列表
    pr_scores: 每次迭代的PR曲线下面积分数列表
    feature_history: 每次迭代添加的特征历史
    """
    print("\n" + "="*80)
    print("特征选择摘要（使用PR曲线下面积作为评估指标）")
    print("="*80)
    
    print("\n前10个最重要的特征:")
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    print(importance_df.head(10).to_string(index=False))
    
    print("\n选中的特征及其重要性:")
    for i, feature_idx in enumerate(selected_features):
        print(f"{i+1}. {feature_names[feature_idx]} (重要性: {feature_importance[feature_idx]:.4f})")
    
    print("\nPR曲线下面积变化趋势:")
    for i, (feature, pr) in enumerate(zip(feature_history, pr_scores)):
        improvement = pr - pr_scores[i-1] if i > 0 else pr
        print(f"添加特征 {i+1}: {feature}, PR曲线下面积: {pr:.4f}, 提升: {improvement:.4f}")
    
    print(f"\n最终PR曲线下面积: {pr_scores[-1]:.4f}")
    print(f"总共选择了 {len(selected_features)} 个特征")

def save_selected_features_dataset(original_file_path, selected_features, feature_names, output_file_path):
    """
    生成并保存包含选定特征的新数据集文件
    
    参数:
    original_file_path: 原始数据文件路径
    selected_features: 选中的特征索引列表
    feature_names: 所有特征的名称列表
    output_file_path: 输出文件路径
    """
    # 读取原始数据
    data = pd.read_csv(original_file_path)
    
    # 获取选中特征的名称
    selected_feature_names = [feature_names[i] for i in selected_features]
    
    # 创建包含选中特征和目标变量的新数据集
    selected_data = data[selected_feature_names + ['Bankrupt']]
    
    # 确保results文件夹存在
    import os
    if not os.path.exists('d:/桌面/学习资料/项目/ml/data'):
        os.makedirs('d:/桌面/学习资料/项目/ml/data')
    
    # 保存新数据集到data文件夹
    results_output_path = f'd:/桌面/学习资料/项目/ml/data/{output_file_path}'
    selected_data.to_csv(results_output_path, index=False)
    
    print(f"\n已保存包含选定特征的新数据集至: {results_output_path}")
    print(f"新数据集包含 {len(selected_feature_names)} 个特征和 {selected_data.shape[0]} 个样本")
    print("选中的特征:")
    for i, feature_name in enumerate(selected_feature_names):
        print(f"{i+1}. {feature_name}")

def process_dataset(file_path, data_type, output_file_path):
    """
    处理单个数据集
    
    参数:
    file_path: 数据文件路径
    data_type: 数据类型，用于区分训练集和测试集
    
    返回:
    selected_features: 选中的特征索引列表
    feature_names: 特征名称列表
    """
    # 加载数据
    print(f"\n处理{data_type}数据集...")
    X, y, feature_names = load_data(file_path)
    print(f"{data_type}数据加载完成，共有 {X.shape[0]} 个样本，{X.shape[1]} 个特征")
    
    # 训练XGBoost模型并提取特征重要性
    print(f"\n训练{data_type}数据集的XGBoost模型...")
    model, feature_importance = train_xgboost_model(X, y)
    print("模型训练完成")
    
    # 前向特征选择
    print(f"\n开始{data_type}数据集的前向特征选择（使用PR曲线下面积作为评估指标）...")
    selected_features, pr_scores, feature_history = forward_feature_selection(X, y, feature_names, feature_importance)
    
    # 打印特征选择摘要
    print(f"\n{data_type}数据集特征选择摘要:")
    print_feature_selection_summary(feature_names, feature_importance, selected_features, pr_scores, feature_history)
    
    # 可视化结果
    print(f"\n生成{data_type}数据集的可视化图表...")
    visualize_results(feature_names, feature_importance, selected_features, pr_scores, feature_history, X, data_type)
    print(f"{data_type}数据集可视化完成")
    
    # 生成并保存包含选定特征的新数据集
    print(f"\n生成{data_type}数据集包含选定特征的新数据集...")
    save_selected_features_dataset(file_path, selected_features, feature_names, output_file_path)
    
    return selected_features, feature_names

def apply_selected_features_to_test_data(test_file_path, train_selected_features, feature_names, output_file_path):
    """
    将训练集选中的特征应用到测试集
    
    参数:
    test_file_path: 测试数据文件路径
    train_selected_features: 训练集选中的特征索引列表
    feature_names: 所有特征的名称列表
    output_file_path: 输出文件路径
    """
    # 读取测试数据
    test_data = pd.read_csv(test_file_path)
    
    # 获取选中特征的名称
    selected_feature_names = [feature_names[i] for i in train_selected_features]
    
    # 创建包含选中特征和目标变量的新数据集
    selected_test_data = test_data[selected_feature_names + ['Bankrupt']]
    
    # 确保data文件夹存在
    import os
    if not os.path.exists('d:/桌面/学习资料/项目/ml/data'):
        os.makedirs('d:/桌面/学习资料/项目/ml/data')
    
    # 保存新数据集到data文件夹
    results_output_path = f'd:/桌面/学习资料/项目/ml/data/{output_file_path}'
    selected_test_data.to_csv(results_output_path, index=False)
    
    print(f"\n已保存测试集选定特征数据至: {results_output_path}")
    print(f"测试集新数据集包含 {len(selected_feature_names)} 个特征和 {selected_test_data.shape[0]} 个样本")
    print("测试集选中的特征:")
    for i, feature_name in enumerate(selected_feature_names):
        print(f"{i+1}. {feature_name}")

'''
# 一次特征选择，刚拿到数据集时
def main():
    # 处理训练数据集
    print("处理训练数据集...")
    train_selected_features, feature_names = process_dataset('d:/桌面/学习资料/项目/ml/data/train_data.csv', 'train', 'selected_train_data.csv')
     
    # 将训练集选中的特征应用到测试集
    print("\n将训练集选中的特征应用到测试集...")
    apply_selected_features_to_test_data('d:/桌面/学习资料/项目/ml/data/test_data.csv', train_selected_features, feature_names, 'selected_test_data.csv')
     
    # 生成测试集的相关性热力图
    print("\n生成测试集的相关性热力图...")
    # 加载测试数据
    test_data = pd.read_csv('d:/桌面/学习资料/项目/ml/data/test_data.csv')
    test_X = test_data.drop('Bankrupt', axis=1).values
    
    # 计算测试集选中特征的相关性
    test_corr_matrix = calculate_feature_correlation(test_X[:, train_selected_features], list(range(len(train_selected_features))))
    
    # 创建热力图
    plt.figure(figsize=(10, 8))
    im = plt.imshow(test_corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    
    # 添加颜色条
    cbar = plt.colorbar(im)
    cbar.set_label('相关系数', fontsize=12)
    
    # 设置刻度标签
    plt.xticks(range(len(train_selected_features)))
    plt.yticks(range(len(train_selected_features)))
    plt.xticks(range(len(train_selected_features)), [feature_names[i] for i in train_selected_features], rotation=45, ha='right')
    plt.yticks(range(len(train_selected_features)), [feature_names[i] for i in train_selected_features])
    
    # 添加标题
    plt.title('测试集选中特征的相关性热力图', fontsize=14, fontweight='bold')
    
    # 在热力图上添加相关系数值
    for i in range(len(train_selected_features)):
        for j in range(len(train_selected_features)):
            text = plt.text(j, i, f'{test_corr_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black" if abs(test_corr_matrix[i, j]) < 0.5 else "white")
    
    # 调整布局
    plt.tight_layout()
    
    # 确保data文件夹存在
    import os
    if not os.path.exists('d:/桌面/学习资料/项目/ml/data'):
        os.makedirs('d:/桌面/学习资料/项目/ml/data')
    
    # 保存图形到data文件夹
    plt.savefig('d:/桌面/学习资料/项目/ml/data/feature_correlation_heatmap_test.png', dpi=300, bbox_inches='tight')
    
    # 显示图形
    plt.show()
    
    print("\n处理完成！")
    print(f"训练集选中的特征数量: {len(train_selected_features)}")
    print("选中的特征:")
    for i, feature_idx in enumerate(train_selected_features):
        print(f"{i+1}. {feature_names[feature_idx]}")
'''
# 二次特征选择，特征衍生之后的特征选择
def main():
    # 处理训练数据集
    print("处理训练数据集...")
    train_selected_features, feature_names = process_dataset('d:/桌面/学习资料/项目/ml/data/derived_train_data.csv', 'train', 'sel_derived_train_data.csv')
     
    # 将训练集选中的特征应用到测试集
    print("\n将训练集选中的特征应用到测试集...")
    apply_selected_features_to_test_data('d:/桌面/学习资料/项目/ml/data/derived_test_data.csv', train_selected_features, feature_names, 'sel_derived_test_data.csv')
     
    # 生成测试集的相关性热力图
    print("\n生成测试集的相关性热力图...")
    # 加载测试数据
    test_data = pd.read_csv('d:/桌面/学习资料/项目/ml/data/derived_test_data.csv')
    test_X = test_data.drop('Bankrupt', axis=1).values
    
    # 计算测试集选中特征的相关性
    test_corr_matrix = calculate_feature_correlation(test_X[:, train_selected_features], list(range(len(train_selected_features))))
    
    # 创建热力图
    plt.figure(figsize=(10, 8))
    im = plt.imshow(test_corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    
    # 添加颜色条
    cbar = plt.colorbar(im)
    cbar.set_label('相关系数', fontsize=12)
    
    # 设置刻度标签
    plt.xticks(range(len(train_selected_features)))
    plt.yticks(range(len(train_selected_features)))
    plt.xticks(range(len(train_selected_features)), [feature_names[i] for i in train_selected_features], rotation=45, ha='right')
    plt.yticks(range(len(train_selected_features)), [feature_names[i] for i in train_selected_features])
    
    # 添加标题
    plt.title('测试集选中特征的相关性热力图', fontsize=14, fontweight='bold')
    
    # 在热力图上添加相关系数值
    for i in range(len(train_selected_features)):
        for j in range(len(train_selected_features)):
            text = plt.text(j, i, f'{test_corr_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black" if abs(test_corr_matrix[i, j]) < 0.5 else "white")
    
    # 调整布局
    plt.tight_layout()
    
    # 确保data文件夹存在
    import os
    if not os.path.exists('d:/桌面/学习资料/项目/ml/data'):
        os.makedirs('d:/桌面/学习资料/项目/ml/data')
    
    # 保存图形到data文件夹
    plt.savefig('d:/桌面/学习资料/项目/ml/data/feature_correlation_heatmap_test.png', dpi=300, bbox_inches='tight')
    
    # 显示图形
    plt.show()
    
    print("\n处理完成！")
    print(f"训练集选中的特征数量: {len(train_selected_features)}")
    print("选中的特征:")
    for i, feature_idx in enumerate(train_selected_features):
        print(f"{i+1}. {feature_names[feature_idx]}")


if __name__ == "__main__":
    main()