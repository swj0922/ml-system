"""
特征分箱分析与优化
实现以下功能：
1. 运用binning_methods中包含的分箱方法，按顺序对数据集中的连续型特征执行分箱处理
2. 将各特征分箱后的模型性能（PR指标）与未分箱时的基准性能进行对比分析
3. 针对每个特征，筛选出能使PR指标提升幅度最大的分箱方法，并从中选定提升效果最为显著的特征进行分箱
4. 依据上述筛选结果，依次对特征实施分箱处理，直至新增的分箱操作不再对PR指标产生提升作用
5. 最终以可视化方式展示各特征的分箱结果及其对应的PR指标提升情况
6. 使用k折交叉验证来选取最优分箱方法
7. 将分箱处理完成后的数据集保存为独立文件
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import average_precision_score, make_scorer
from binning_methods import ChiSquaredBinning, BestKSBinning, DecisionTreeBinning
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 自定义PR AUC评分器
pr_auc_scorer = make_scorer(average_precision_score, needs_proba=True)

class FeatureBinningAnalyzer:
    """
    特征分箱分析器
    用于分析不同分箱方法对模型性能的影响，并选择最优分箱策略
    """
    
    def __init__(self, train_data_path, test_data_path, target_column='Bankrupt', random_state=42, cv_folds=5):
        """
        初始化特征分箱分析器
        
        参数:
        train_data_path: 训练数据文件路径
        test_data_path: 测试数据文件路径
        target_column: 目标列名
        random_state: 随机种子
        cv_folds: 交叉验证折数
        """
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.target_column = target_column
        self.random_state = random_state
        self.cv_folds = cv_folds
        
        # 加载训练数据和测试数据
        self.train_data = pd.read_csv(train_data_path)
        self.test_data = pd.read_csv(test_data_path)
        
        # 分离特征和目标变量 - 训练集
        self.X_train = self.train_data.drop(columns=[target_column])
        self.y_train = self.train_data[target_column]
        
        # 分离特征和目标变量 - 测试集
        self.X_test = self.test_data.drop(columns=[target_column])
        self.y_test = self.test_data[target_column]
        
        # 创建交叉验证对象
        self.cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
        # 存储特征名
        self.feature_names = self.X_train.columns.tolist()
        
        # 存储分箱方法
        self.binning_methods = {
            'ChiSquared': ChiSquaredBinning,
            'BestKS': BestKSBinning,
            'DecisionTree': DecisionTreeBinning
        }
        
        # 存储基准性能（未分箱）
        self.baseline_score = self._evaluate_model_cv(self.X_train, self.y_train)
        
        # 存储各特征分箱后的性能
        self.feature_binning_scores = {}
        
        # 存储最优分箱方法
        self.best_binning_methods = {}
        
        # 存储最终选择的特征及其分箱方法
        self.selected_features = []
        
        # 存储逐步分箱的性能变化
        self.stepwise_scores = [self.baseline_score]
        
        # 存储分箱后的数据集
        self.binned_train_data = None
        self.binned_test_data = None
        
    def _evaluate_model_cv(self, X, y):
        """
        使用k折交叉验证评估模型性能（PR指标）
        
        参数:
        X: 特征数据
        y: 目标变量
        
        返回:
        平均PR AUC分数
        """
        # 使用XGBoost模型
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
        
        # 使用交叉验证计算PR AUC
        cv_scores = []
        for train_idx, val_idx in self.cv.split(X, y):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            # 训练模型
            model.fit(X_train_fold, y_train_fold)
            
            # 预测概率
            y_scores = model.predict_proba(X_val_fold)[:, 1]
            
            # 计算PR AUC
            pr_auc = average_precision_score(y_val_fold, y_scores)
            cv_scores.append(pr_auc)
        
        # 返回平均分数
        return np.mean(cv_scores)
    
    def analyze_feature_binning(self, max_bins=10):
        """
        分析每个特征的分箱效果
        
        参数:
        max_bins: 最大分箱数
        
        说明:
        对每个特征单独进行分箱处理，其他特征保持不变，然后计算PR指标
        """
        print(f"基准性能（未分箱）PR AUC: {self.baseline_score:.4f}")
        
        # 对每个特征进行分析
        for feature in self.feature_names:
            print(f"\n分析特征: {feature}")
            print(f"仅对特征 '{feature}' 进行分箱处理，其他特征保持不变")
            
            feature_scores = {}
            
            # 尝试每种分箱方法
            for method_name, method_class in self.binning_methods.items():
                # 创建分箱器
                binner = method_class(max_bins=max_bins)
                
                # 仅对当前特征进行分箱处理
                train_binned = binner.fit_transform(self.X_train[feature].values, self.y_train)
                
                # 创建副本以避免修改原始数据
                X_train_binned = self.X_train.copy()
                # 只替换当前特征的分箱值，其他特征保持不变
                X_train_binned[feature] = train_binned
                
                # 使用交叉验证评估性能
                score = self._evaluate_model_cv(X_train_binned, self.y_train)
                feature_scores[method_name] = score
                
                # 计算提升幅度
                improvement = score - self.baseline_score
                print(f"  {method_name}分箱: PR AUC = {score:.4f}, 提升 = {improvement:.4f}")
            
            # 存储该特征的分箱结果
            self.feature_binning_scores[feature] = feature_scores
            
            # 选择最佳分箱方法
            best_method = max(feature_scores, key=feature_scores.get)
            best_score = feature_scores[best_method]
            best_improvement = best_score - self.baseline_score
            
            self.best_binning_methods[feature] = {
                'method': best_method,
                'score': best_score,
                'improvement': best_improvement
            }
            
            print(f"  最佳分箱方法: {best_method}, 提升 = {best_improvement:.4f}")
    
    def select_features_sequentially(self):
        """
        逐步选择特征进行分箱，直到连续三轮都没有提升，然后回溯到之前的最优结果
        """
        # 按提升幅度排序特征
        sorted_features = sorted(
            self.best_binning_methods.items(),
            key=lambda x: x[1]['improvement'],
            reverse=True
        )
        
        # 初始化数据副本
        X_train_current = self.X_train.copy()
        X_test_current = self.X_test.copy()
        
        # 记录最优结果
        best_score = self.baseline_score
        best_selected_features = []
        best_X_train = X_train_current.copy()
        best_X_test = X_test_current.copy()
        
        # 记录连续没有提升的轮数
        no_improvement_count = 0
        
        # 逐步添加分箱特征
        for feature, info in sorted_features:
            # 如果连续三轮没有提升，则回溯到最优结果并停止
            if no_improvement_count >= 3:
                print(f"\n连续三轮没有提升，回溯到之前的最优结果")
                self.selected_features = best_selected_features.copy()
                self.stepwise_scores = [self.baseline_score] + [f_info['score'] for f_info in best_selected_features]
                print(f"最终选择了 {len(self.selected_features)} 个特征进行分箱")
                break
                
            print(f"\n尝试对特征 {feature} 使用 {info['method']} 分箱")
            
            # 获取分箱方法类
            method_class = self.binning_methods[info['method']]
            
            # 创建分箱器
            binner = method_class(max_bins=10)
            
            # 对训练集进行分箱
            train_binned = binner.fit_transform(X_train_current[feature].values, self.y_train)
            
            # 对测试集进行分箱
            test_binned = binner.transform(X_test_current[feature].values)
            
            # 创建副本以避免修改原始数据
            X_train_binned = X_train_current.copy()
            X_test_binned = X_test_current.copy()
            
            # 替换特征值为分箱后的值
            X_train_binned[feature] = train_binned
            X_test_binned[feature] = test_binned
            
            # 使用交叉验证评估性能
            current_score = self._evaluate_model_cv(X_train_binned, self.y_train)
            
            # 计算相对于上一步的提升
            previous_score = self.stepwise_scores[-1]
            step_improvement = current_score - previous_score
            
            print(f"当前性能: PR AUC = {current_score:.4f}")
            print(f"相对于上一步的提升: {step_improvement:.4f}")
            
            # 如果有提升，则保留该分箱
            if step_improvement > 0:
                self.selected_features.append({
                    'feature': feature,
                    'method': info['method'],
                    'score': current_score,
                    'improvement': step_improvement,
                    'binner': binner  # 保存分箱器以便后续使用
                })
                
                # 更新当前数据
                X_train_current = X_train_binned
                X_test_current = X_test_binned
                
                # 记录性能
                self.stepwise_scores.append(current_score)
                
                # 如果当前性能优于最优性能，则更新最优结果
                if current_score > best_score:
                    best_score = current_score
                    best_selected_features = self.selected_features.copy()
                    best_X_train = X_train_current.copy()
                    best_X_test = X_test_current.copy()
                    print(f"更新最优结果，当前最优PR AUC: {best_score:.4f}")
                
                # 重置没有提升的计数器
                no_improvement_count = 0
                
                print(f"保留特征 {feature} 的分箱")
            else:
                # 增加没有提升的计数器
                no_improvement_count += 1
                print(f"特征 {feature} 的分箱不能提升性能，连续 {no_improvement_count} 轮没有提升")
        
        # 如果遍历完所有特征但连续没有提升的轮数不足3轮，则使用当前结果
        if no_improvement_count < 3:
            print(f"\n已尝试所有特征，最终选择了 {len(self.selected_features)} 个特征进行分箱")
        
        # 保存分箱后的完整数据集
        self._save_binned_dataset()
    
    def _save_binned_dataset(self):
        """
        保存分箱后的训练集和测试集
        """
        # 创建训练数据的副本
        binned_train_data = self.train_data.copy()
        
        # 创建测试数据的副本
        binned_test_data = self.test_data.copy()
        
        # 对每个选定的特征应用分箱
        for info in self.selected_features:
            feature = info['feature']
            binner = info['binner']
            
            # 对训练集进行分箱
            binned_train_values = binner.transform(self.X_train[feature].values)
            binned_train_data[feature] = binned_train_values
            
            # 对测试集进行分箱
            binned_test_values = binner.transform(self.X_test[feature].values)
            binned_test_data[feature] = binned_test_values
        
        # 保存分箱后的训练集
        binned_train_data.to_csv('data/binned_train_data.csv', index=False)
        self.binned_train_data = binned_train_data
        
        # 保存分箱后的测试集
        binned_test_data.to_csv('data/binned_test_data.csv', index=False)
        self.binned_test_data = binned_test_data
        
        print("\n分箱后的训练集已保存为 'data/binned_train_data.csv'")
        print("分箱后的测试集已保存为 'data/binned_test_data.csv'")
    
    def visualize_results(self):
        """
        可视化分箱结果和性能提升
        """
        # 创建图形
        plt.figure(figsize=(15, 10))
        
        # 1. 各特征不同分箱方法的性能对比
        plt.subplot(2, 2, 1)
        
        # 准备数据
        features = list(self.feature_binning_scores.keys())
        methods = list(self.binning_methods.keys())
        
        # 创建DataFrame
        data = []
        for feature in features:
            for method in methods:
                score = self.feature_binning_scores[feature][method]
                improvement = score - self.baseline_score
                data.append({
                    'Feature': feature,
                    'Method': method,
                    'Improvement': improvement
                })
        
        df = pd.DataFrame(data)
        
        # 绘制条形图
        sns.barplot(x='Improvement', y='Feature', hue='Method', data=df)
        plt.title('各特征不同分箱方法的PR AUC提升')
        plt.xlabel('PR AUC提升幅度')
        plt.ylabel('特征')
        plt.axvline(x=0, color='r', linestyle='--')
        
        # 2. 最佳分箱方法的提升幅度
        plt.subplot(2, 2, 2)
        
        # 准备数据
        best_data = []
        for feature, info in self.best_binning_methods.items():
            best_data.append({
                'Feature': feature,
                'Method': info['method'],
                'Improvement': info['improvement']
            })
        
        best_df = pd.DataFrame(best_data)
        best_df = best_df.sort_values('Improvement', ascending=False)
        
        # 绘制条形图
        sns.barplot(x='Improvement', y='Feature', data=best_df, palette='viridis')
        plt.title('各特征最佳分箱方法的PR AUC提升')
        plt.xlabel('PR AUC提升幅度')
        plt.ylabel('特征')
        plt.axvline(x=0, color='r', linestyle='--')
        
        # 3. 逐步分箱的性能变化
        plt.subplot(2, 2, 3)
        
        # 准备数据
        steps = list(range(len(self.stepwise_scores)))
        step_labels = ['基准'] + [f"{info['feature']}({info['method']})" for info in self.selected_features]
        
        # 绘制折线图
        plt.plot(steps, self.stepwise_scores, marker='o', linestyle='-')
        plt.xticks(steps, step_labels, rotation=45, ha='right')
        plt.title('逐步分箱的PR AUC变化')
        plt.xlabel('分箱步骤')
        plt.ylabel('PR AUC')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 4. 选定特征的分箱分布
        if self.selected_features:
            plt.subplot(2, 2, 4)
            
            # 选择第一个选定的特征进行可视化
            first_selected = self.selected_features[0]
            feature = first_selected['feature']
            method = first_selected['method']
            
            # 获取分箱方法类
            method_class = self.binning_methods[method]
            
            # 创建分箱器
            binner = method_class(max_bins=10)
            
            # 对训练集进行分箱
            train_binned = binner.fit_transform(self.X_train[feature].values, self.y_train)
            
            # 创建DataFrame
            bin_df = pd.DataFrame({
                'Feature': self.X_train[feature].values,
                'Bin': train_binned,
                'Target': self.y_train.values
            })
            
            # 计算每个分箱的目标变量均值
            bin_stats = bin_df.groupby('Bin')['Target'].agg(['mean', 'count']).reset_index()
            
            # 绘制条形图
            sns.barplot(x='Bin', y='mean', data=bin_stats)
            plt.title(f'特征 {feature} 使用 {method} 分箱后的目标变量均值')
            plt.xlabel('分箱')
            plt.ylabel('目标变量均值（破产概率）')
            
            # 添加样本量标签
            for i, row in bin_stats.iterrows():
                plt.text(i, row['mean'] + 0.01, f"n={row['count']}", ha='center')
        
        plt.tight_layout()
        plt.savefig('feature_binning_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("\n分析结果已保存为 'feature_binning_analysis.png'")
    
    def run_analysis(self, max_bins=10, cv_folds=5):
        """
        运行完整的分析流程
        
        参数:
        max_bins: 最大分箱数
        cv_folds: 交叉验证折数
        """
        print("开始特征分箱分析...")
        print(f"使用 {cv_folds} 折交叉验证")
        
        # 1. 分析每个特征的分箱效果
        print("\n步骤1: 分析每个特征的分箱效果")
        self.analyze_feature_binning(max_bins)
        
        # 2. 逐步选择特征进行分箱
        print("\n步骤2: 逐步选择特征进行分箱")
        self.select_features_sequentially()
        
        # 3. 可视化结果
        print("\n步骤3: 可视化结果")
        self.visualize_results()
        
        # 输出总结
        print("\n分析总结:")
        print(f"基准性能（未分箱）PR AUC: {self.baseline_score:.4f}")
        print(f"最终性能（分箱后）PR AUC: {self.stepwise_scores[-1]:.4f}")
        print(f"总体提升: {self.stepwise_scores[-1] - self.baseline_score:.4f}")
        
        print("\n选定的分箱特征:")
        for info in self.selected_features:
            print(f"特征: {info['feature']}, 方法: {info['method']}, 提升: {info['improvement']:.4f}")
        
        print("\n分箱后的训练集已保存为 'data/binned_train_data.csv'")
        print("分箱后的测试集已保存为 'data/binned_test_data.csv'")


# 主程序
if __name__ == "__main__":
    # 创建分析器，使用预先划分的训练数据和测试数据，5折交叉验证
    analyzer = FeatureBinningAnalyzer('data/sel_derived_train_data.csv', 'data/sel_derived_test_data.csv', cv_folds=5)
    
    # 运行分析
    analyzer.run_analysis(max_bins=10, cv_folds=5)