"""
机器学习分箱方法实现
包含三种分箱方法：
1. 卡方分箱（Chi-squared Binning）
2. 最佳KS分箱（Best KS Binning）
3. 决策树分箱（Decision Tree Binning）
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class ChiSquaredBinning:
    """
    卡方分箱（Chi-squared Binning）
    基于卡方检验的合并策略，将相似的分箱合并
    """
    
    def __init__(self, max_bins=10, min_bin_size=0.05, confidence_level=0.95):
        """
        初始化卡方分箱器
        
        参数:
        max_bins: 最大分箱数
        min_bin_size: 最小分箱占比（防止过小的分箱）
        confidence_level: 卡方检验的置信水平
        """
        self.max_bins = max_bins
        self.min_bin_size = min_bin_size
        self.confidence_level = confidence_level
        self.bins = None
        self.chi2_values = None
        
    def fit(self, X, y):
        """
        训练卡方分箱器
        
        参数:
        X: 特征值（一维数组）
        y: 目标变量（0/1）
        """
        # 将特征值和目标变量合并为DataFrame
        df = pd.DataFrame({'X': X, 'y': y})
        
        # 初始分箱：使用分位数
        df_sorted = df.sort_values('X')
        percentiles = np.linspace(0, 100, min(20, len(X)))  # 最多20个初始分箱
        initial_bins = np.percentile(df_sorted['X'], percentiles[1:-1])
        
        # 去除重复的分箱边界
        bins = np.unique(initial_bins)
        
        # 如果唯一值太少，直接返回
        if len(bins) <= self.max_bins - 1:
            self.bins = bins
            return self
        
        # 计算每个分箱的正负样本数
        def calculate_bins_stats(bins):
            bin_stats = []
            for i in range(len(bins) + 1):
                if i == 0:
                    bin_data = df_sorted[df_sorted['X'] < bins[0]]
                elif i == len(bins):
                    bin_data = df_sorted[df_sorted['X'] >= bins[-1]]
                else:
                    bin_data = df_sorted[(df_sorted['X'] >= bins[i-1]) & (df_sorted['X'] < bins[i])]
                
                pos_count = sum(bin_data['y'])
                neg_count = len(bin_data) - pos_count
                bin_stats.append((pos_count, neg_count))
            return bin_stats
        
        # 计算卡方值
        def calculate_chi2(bin1, bin2):
            pos1, neg1 = bin1
            pos2, neg2 = bin2
            
            total_pos = pos1 + pos2
            total_neg = neg1 + neg2
            total = total_pos + total_neg
            
            if total == 0:
                return float('inf')  # 返回无穷大，避免合并空分箱
            
            # 期望值
            exp_pos1 = total_pos * (pos1 + neg1) / total
            exp_neg1 = total_neg * (pos1 + neg1) / total
            exp_pos2 = total_pos * (pos2 + neg2) / total
            exp_neg2 = total_neg * (pos2 + neg2) / total
            
            # 避免除以零
            if exp_pos1 == 0 or exp_neg1 == 0 or exp_pos2 == 0 or exp_neg2 == 0:
                return float('inf')  # 返回无穷大，避免合并可能导致除零的分箱
            
            # 卡方值
            chi2 = ((pos1 - exp_pos1)**2 / exp_pos1 + 
                    (neg1 - exp_neg1)**2 / exp_neg1 + 
                    (pos2 - exp_pos2)**2 / exp_pos2 + 
                    (neg2 - exp_neg2)**2 / exp_neg2)
            
            return chi2
        
        # 卡方阈值
        chi2_threshold = stats.chi2.ppf(self.confidence_level, df=1)
        
        # 初始分箱统计
        bin_stats = calculate_bins_stats(bins)
        
        # 不断合并相邻分箱，直到达到停止条件
        while len(bins) > self.max_bins - 1:  # max_bins-1个边界会产生max_bins个分箱
            # 计算相邻分箱的卡方值
            chi2_values = []
            for i in range(len(bin_stats) - 1):
                chi2 = calculate_chi2(bin_stats[i], bin_stats[i+1])
                chi2_values.append(chi2)
            
            # 找到卡方值最小的相邻分箱对
            min_chi2_idx = np.argmin(chi2_values)
            min_chi2 = chi2_values[min_chi2_idx]
            
            # 如果最小卡方值大于阈值，停止合并
            if min_chi2 > chi2_threshold:
                # 如果因为卡方阈值而停止合并，但分箱数仍然超过max_bins-1，强制合并
                while len(bins) > self.max_bins - 1:
                    # 找到样本量最小的分箱进行合并
                    bin_sizes = [bin_stats[i][0] + bin_stats[i][1] for i in range(len(bin_stats))]
                    min_size_idx = np.argmin(bin_sizes)
                    
                    # 合并分箱
                    if min_size_idx > 0:
                        # 与前一个分箱合并
                        pos1, neg1 = bin_stats[min_size_idx-1]
                        pos2, neg2 = bin_stats[min_size_idx]
                        bin_stats[min_size_idx-1] = (pos1 + pos2, neg1 + neg2)
                        bin_stats.pop(min_size_idx)
                        bins = np.delete(bins, min_size_idx-1)
                    else:
                        # 与后一个分箱合并
                        pos1, neg1 = bin_stats[min_size_idx]
                        pos2, neg2 = bin_stats[min_size_idx+1]
                        bin_stats[min_size_idx] = (pos1 + pos2, neg1 + neg2)
                        bin_stats.pop(min_size_idx+1)
                        bins = np.delete(bins, min_size_idx)
                break
            
            # 合并分箱
            pos1, neg1 = bin_stats[min_chi2_idx]
            pos2, neg2 = bin_stats[min_chi2_idx + 1]
            bin_stats[min_chi2_idx] = (pos1 + pos2, neg1 + neg2)
            bin_stats.pop(min_chi2_idx + 1)
            
            # 更新分箱边界
            bins = np.delete(bins, min_chi2_idx)
            
            # 如果已经达到最大分箱数，强制停止
            if len(bins) <= self.max_bins - 1:
                break
        
        self.bins = bins
        self.chi2_values = chi2_values
        return self
    
    def transform(self, X):
        """
        应用卡方分箱
        
        参数:
        X: 特征值（一维数组）
        
        返回:
        分箱后的特征值
        """
        if self.bins is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 使用digitize进行分箱
        bins_with_edges = np.concatenate([[-np.inf], self.bins, [np.inf]])
        binned_X = np.digitize(X, bins_with_edges) - 1
        
        return binned_X
    
    def fit_transform(self, X, y):
        """
        训练并应用卡方分箱
        """
        self.fit(X, y)
        return self.transform(X)


class BestKSBinning:
    """
    最佳KS分箱（Best KS Binning）
    基于KS统计量的最优分箱方法，最大化区分正负样本的能力
    """
    
    def __init__(self, max_bins=10, min_bin_size=0.05):
        """
        初始化最佳KS分箱器
        
        参数:
        max_bins: 最大分箱数
        min_bin_size: 最小分箱占比（防止过小的分箱）
        """
        self.max_bins = max_bins
        self.min_bin_size = min_bin_size
        self.bins = None
        self.ks_values = None
        
    def fit(self, X, y):
        """
        训练最佳KS分箱器
        
        参数:
        X: 特征值（一维数组）
        y: 目标变量（0/1）
        """
        # 将特征值和目标变量合并为DataFrame
        df = pd.DataFrame({'X': X, 'y': y})
        
        # 按特征值排序
        df_sorted = df.sort_values('X')
        
        # 计算KS值
        def calculate_ks(split_point):
            # 左右分箱
            left_bin = df_sorted[df_sorted['X'] <= split_point]
            right_bin = df_sorted[df_sorted['X'] > split_point]
            
            if len(left_bin) == 0 or len(right_bin) == 0:
                return 0
            
            # 计算累积分布
            left_pos_rate = sum(left_bin['y']) / len(left_bin)
            left_neg_rate = (len(left_bin) - sum(left_bin['y'])) / len(left_bin)
            
            right_pos_rate = sum(right_bin['y']) / len(right_bin)
            right_neg_rate = (len(right_bin) - sum(right_bin['y'])) / len(right_bin)
            
            # KS值
            ks = abs(left_pos_rate - right_pos_rate) + abs(left_neg_rate - right_neg_rate)
            return ks
        
        # 获取所有可能的分割点
        unique_values = df_sorted['X'].unique()
        
        # 如果唯一值太少，直接返回
        if len(unique_values) <= self.max_bins - 1:
            self.bins = np.sort(unique_values)
            return self
        
        # 初始分箱：使用分位数
        percentiles = np.linspace(0, 100, self.max_bins + 1)
        initial_bins = np.percentile(X, percentiles[1:-1])
        
        # 去除重复的分箱边界
        initial_bins = np.unique(initial_bins)
        
        # 如果初始分箱太少，使用均匀间隔
        if len(initial_bins) < self.max_bins - 1:
            initial_bins = np.linspace(np.min(X), np.max(X), self.max_bins + 1)[1:-1]
        
        # 优化分箱
        bins = initial_bins.copy()
        ks_values = []
        
        # 对每个分箱边界进行优化
        for iteration in range(10):  # 最多迭代10次
            new_bins = []
            new_ks_values = []
            
            # 对每个分箱边界寻找最佳分割点
            for i in range(len(bins)):
                # 在当前边界附近搜索最佳分割点
                left_bound = np.min(X) if i == 0 else bins[i-1]
                right_bound = np.max(X) if i == len(bins)-1 else bins[i+1]
                
                # 确保左右边界不同
                if left_bound == right_bound:
                    new_bins.append(left_bound)
                    new_ks_values.append(0)
                    continue
                
                # 在边界附近取一些候选点
                candidates = np.linspace(left_bound, right_bound, 20)[1:-1]
                
                # 确保候选点不重复
                candidates = np.unique(candidates)
                
                if len(candidates) == 0:
                    new_bins.append((left_bound + right_bound) / 2)
                    new_ks_values.append(0)
                    continue
                
                # 计算每个候选点的KS值
                candidate_ks = [calculate_ks(candidate) for candidate in candidates]
                
                # 选择KS值最大的候选点
                best_idx = np.argmax(candidate_ks)
                best_candidate = candidates[best_idx]
                best_ks = candidate_ks[best_idx]
                
                new_bins.append(best_candidate)
                new_ks_values.append(best_ks)
            
            # 检查是否收敛
            if np.allclose(bins, new_bins, rtol=1e-3):
                break
                
            bins = np.array(new_bins)
            ks_values = new_ks_values
        
        # 确保分箱边界是唯一的
        bins = np.unique(bins)
        
        # 如果分箱数仍然超过max_bins-1，选择KS值最大的边界
        if len(bins) > self.max_bins - 1:
            # 计算每个边界的KS值
            bin_ks_values = [calculate_ks(bin_val) for bin_val in bins]
            
            # 选择KS值最大的max_bins-1个边界
            top_indices = np.argsort(bin_ks_values)[-(self.max_bins-1):]
            bins = sorted([bins[i] for i in top_indices])
        
        self.bins = bins
        self.ks_values = ks_values
        return self
    
    def transform(self, X):
        """
        应用最佳KS分箱
        
        参数:
        X: 特征值（一维数组）
        
        返回:
        分箱后的特征值
        """
        if self.bins is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 使用digitize进行分箱
        bins_with_edges = np.concatenate([[-np.inf], self.bins, [np.inf]])
        binned_X = np.digitize(X, bins_with_edges) - 1
        
        return binned_X
    
    def fit_transform(self, X, y):
        """
        训练并应用最佳KS分箱
        """
        self.fit(X, y)
        return self.transform(X)


class DecisionTreeBinning:
    """
    决策树分箱（Decision Tree Binning）
    使用决策树找到最优分割点
    """
    
    def __init__(self, max_bins=10, min_bin_size=0.05, random_state=None):
        """
        初始化决策树分箱器
        
        参数:
        max_bins: 最大分箱数
        min_bin_size: 最小分箱占比（防止过小的分箱）
        random_state: 随机种子
        """
        self.max_bins = max_bins
        self.min_bin_size = min_bin_size
        self.random_state = random_state
        self.bins = None
        self.tree = None
        
    def fit(self, X, y):
        """
        训练决策树分箱器
        
        参数:
        X: 特征值（一维数组）
        y: 目标变量（0/1）
        """
        # 转换为二维数组，因为sklearn需要二维输入
        X_2d = X.reshape(-1, 1)
        
        # 创建决策树
        # 限制树的深度，使得分箱数不超过max_bins
        max_depth = min(int(np.log2(self.max_bins)) + 1, 10)
        min_samples_leaf = max(int(len(X) * self.min_bin_size), 1)
        
        tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=self.random_state
        )
        
        # 训练决策树
        tree.fit(X_2d, y)
        
        # 获取决策树的分割点
        def get_tree_bins(tree, feature_idx=0):
            # 获取树的分割点
            tree_ = tree.tree_
            feature_name = tree_.feature
            
            # 递归获取分割点
            def recurse(node, depth):
                if tree_.feature[node] != feature_idx:
                    return []
                
                if tree_.children_left[node] == tree_.children_right[node]:
                    return []
                
                threshold = tree_.threshold[node]
                left_bins = recurse(tree_.children_left[node], depth + 1)
                right_bins = recurse(tree_.children_right[node], depth + 1)
                
                return [threshold] + left_bins + right_bins
            
            bins = recurse(0, 0)
            return sorted(bins)
        
        # 获取分箱边界
        bins = get_tree_bins(tree)
        
        # 如果分箱数超过max_bins，选择最重要的分割点
        if len(bins) > self.max_bins:
            # 计算每个分割点的重要性
            feature_importance = tree.feature_importances_[0]
            thresholds = tree.tree_.threshold[tree.tree_.feature == 0]
            importances = [feature_importance] * len(thresholds)
            
            # 选择重要性最高的max_bins-1个分割点
            if len(thresholds) > self.max_bins - 1:
                top_indices = np.argsort(importances)[-(self.max_bins-1):]
                bins = sorted([thresholds[i] for i in top_indices])
            else:
                bins = sorted(thresholds)
        
        self.bins = bins
        self.tree = tree
        return self
    
    def transform(self, X):
        """
        应用决策树分箱
        
        参数:
        X: 特征值（一维数组）
        
        返回:
        分箱后的特征值
        """
        if self.bins is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 使用digitize进行分箱
        bins_with_edges = np.concatenate([[-np.inf], self.bins, [np.inf]])
        binned_X = np.digitize(X, bins_with_edges) - 1
        
        return binned_X
    
    def fit_transform(self, X, y):
        """
        训练并应用决策树分箱
        """
        self.fit(X, y)
        return self.transform(X)


def compare_binning_methods(X, y, max_bins=5):
    """
    比较三种分箱方法的效果
    
    参数:
    X: 特征值（一维数组）
    y: 目标变量（0/1）
    max_bins: 最大分箱数
    
    返回:
    包含三种分箱结果的DataFrame
    """
    # 初始化三种分箱方法
    chi2_binning = ChiSquaredBinning(max_bins=max_bins)
    ks_binning = BestKSBinning(max_bins=max_bins)
    dt_binning = DecisionTreeBinning(max_bins=max_bins)
    
    # 应用分箱
    X_chi2 = chi2_binning.fit_transform(X, y)
    X_ks = ks_binning.fit_transform(X, y)
    X_dt = dt_binning.fit_transform(X, y)
    
    # 创建结果DataFrame
    result = pd.DataFrame({
        'X': X,
        'y': y,
        'Chi2_Bin': X_chi2,
        'KS_Bin': X_ks,
        'DT_Bin': X_dt
    })
    
    # 计算每个分箱的正样本比例
    chi2_bin_stats = result.groupby('Chi2_Bin')['y'].agg(['mean', 'count']).reset_index()
    ks_bin_stats = result.groupby('KS_Bin')['y'].agg(['mean', 'count']).reset_index()
    dt_bin_stats = result.groupby('DT_Bin')['y'].agg(['mean', 'count']).reset_index()
    
    return {
        'binned_data': result,
        'chi2_bins': chi2_binning.bins,
        'ks_bins': ks_binning.bins,
        'dt_bins': dt_binning.bins,
        'chi2_bin_stats': chi2_bin_stats,
        'ks_bin_stats': ks_bin_stats,
        'dt_bin_stats': dt_bin_stats
    }


# 示例使用
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    n_samples = 10000
    
    # 生成特征值
    X = np.random.normal(0, 1, n_samples)
    
    # 生成目标变量（与X相关）
    prob = 1 / (1 + np.exp(-X))  # sigmoid函数
    y = np.random.binomial(1, prob)
    
    # 比较三种分箱方法
    results = compare_binning_methods(X, y, max_bins=5)
    
    # 打印分箱边界
    print("卡方分箱边界:", results['chi2_bins'])
    print("KS分箱边界:", results['ks_bins'])
    print("决策树分箱边界:", results['dt_bins'])
    
    # 打印分箱统计信息
    print("\n卡方分箱统计:")
    print(results['chi2_bin_stats'])
    
    print("\nKS分箱统计:")
    print(results['ks_bin_stats'])
    
    print("\n决策树分箱统计:")
    print(results['dt_bin_stats'])
    
    # 打印部分分箱结果
    print("\n分箱结果示例:")
    print(results['binned_data'].head(10))