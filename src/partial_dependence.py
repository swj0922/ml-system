"""
部分依赖图(Partial Dependence Plot)计算模块
实现机器学习模型的部分依赖图计算和可视化功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import seaborn as sns
from typing import List, Dict, Any, Tuple, Optional, Union
import base64
import io
import json
from sklearn.inspection import partial_dependence
import joblib

# 设置中文字体
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

class PartialDependenceAnalyzer:
    """部分依赖图分析器"""
    
    def __init__(self, model, feature_names: List[str], train_data: pd.DataFrame):
        """
        初始化部分依赖图分析器
        
        Args:
            model: 训练好的机器学习模型
            feature_names: 特征名称列表
            train_data: 训练数据
        """
        self.model = model
        self.feature_names = feature_names
        self.train_data = train_data
        self.X_train = train_data[feature_names]
        
    def calculate_partial_dependence(self, 
                                   features: List[str], 
                                   grid_resolution: int = 100,
                                   percentiles: Tuple[float, float] = (0.05, 0.95)) -> Dict[str, Any]:
        """
        计算指定特征的部分依赖
        
        Args:
            features: 要计算部分依赖的特征列表
            grid_resolution: 网格分辨率
            percentiles: 特征值的百分位数范围
            
        Returns:
            包含部分依赖数据的字典
        """
        results = {}
        
        for feature in features:
            if feature not in self.feature_names:
                continue
                
            feature_idx = self.feature_names.index(feature)
            
            # 计算特征值范围
            feature_values = self.X_train[feature]
            min_val = np.percentile(feature_values, percentiles[0] * 100)
            max_val = np.percentile(feature_values, percentiles[1] * 100)
            
            # 创建网格点
            grid_values = np.linspace(min_val, max_val, grid_resolution)
            
            # 计算部分依赖
            try:
                pd_result = partial_dependence(
                    self.model, 
                    self.X_train, 
                    features=[feature_idx],
                    grid_resolution=grid_resolution,
                    percentiles=percentiles
                )
                
                # 提取结果
                partial_dependence_values = pd_result['average'][0]
                grid_values = pd_result['grid_values'][0]
                
                results[feature] = {
                    'grid_values': grid_values.tolist(),
                    'partial_dependence': partial_dependence_values.tolist(),
                    'feature_range': [float(min_val), float(max_val)],
                    'feature_stats': {
                        'mean': float(feature_values.mean()),
                        'std': float(feature_values.std()),
                        'min': float(feature_values.min()),
                        'max': float(feature_values.max()),
                        'median': float(feature_values.median())
                    }
                }
                
            except Exception as e:
                print(f"计算特征 {feature} 的部分依赖时出错: {e}")
                continue
                
        return results
    
    def create_pdp_plots(self, 
                        features: List[str], 
                        sample_data: Optional[Dict[str, float]] = None,
                        grid_resolution: int = 100,
                        figsize: Tuple[int, int] = (15, 10),
                        pdp_data: Optional[Dict[str, Any]] = None) -> str:
        """
        创建部分依赖图
        
        Args:
            features: 要绘制的特征列表（最多9个）
            sample_data: 可选的样本数据，用于在图上标注
            grid_resolution: 网格分辨率
            figsize: 图形大小
            pdp_data: 可选的预计算PDP数据，如果提供则不重新计算
            
        Returns:
            Base64编码的图像字符串
        """
        # 限制最多9个特征
        features = features[:9]
        
        # 使用预计算的数据或重新计算部分依赖
        if pdp_data is None:
            pdp_data = self.calculate_partial_dependence(features, grid_resolution)
        else:
            # 过滤出需要的特征数据
            pdp_data = {feature: pdp_data[feature] for feature in features if feature in pdp_data}
        
        if not pdp_data:
            raise ValueError("无法计算任何特征的部分依赖")
        
        # 计算子图布局
        n_features = len(pdp_data)
        if n_features <= 3:
            rows, cols = 1, n_features
        elif n_features <= 6:
            rows, cols = 2, 3
        else:
            rows, cols = 3, 3
            
        # 创建图形
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if n_features == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        # 设置样式
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # 定义统一的绿色方案
        green_colors = [
            '#2ca02c',  # 标准绿色
            '#228B22',  # 森林绿
            '#32CD32',  # 酸橙绿
            '#00FF7F',  # 春绿色
            '#90EE90',  # 浅绿色
            '#006400',  # 深绿色
            '#7CFC00',  # 草坪绿
            '#00FA9A',  # 中春绿
            '#98FB98'   # 淡绿色
        ]
        
        # 确保有足够的绿色变体
        colors = green_colors[:n_features] if n_features <= len(green_colors) else ['#2ca02c'] * n_features
        
        for idx, (feature, data) in enumerate(pdp_data.items()):
            ax = axes[idx]
            
            # 绘制部分依赖曲线
            ax.plot(data['grid_values'], data['partial_dependence'], 
                   color=colors[idx], linewidth=3.0, alpha=0.9, 
                   marker='o', markersize=2, markevery=max(1, len(data['grid_values'])//20))
            
            # 填充曲线下方区域
            ax.fill_between(data['grid_values'], data['partial_dependence'], 
                           alpha=0.2, color=colors[idx])
            
            # 如果有样本数据，标注样本点
            if sample_data and feature in sample_data:
                sample_value = sample_data[feature]
                
                # 插值计算样本点对应的部分依赖值
                interp_pd = np.interp(sample_value, data['grid_values'], data['partial_dependence'])
                
                # 绘制垂直线
                ax.axvline(x=sample_value, color='#FF4444', linestyle='--', 
                          linewidth=2.5, alpha=0.9, label=f'sample value: {sample_value:.4f}')
                
                # 绘制样本点
                ax.scatter([sample_value], [interp_pd], color='#FF4444', s=120, 
                          zorder=5, edgecolors='#CC0000', linewidth=2.5, alpha=0.9)
                
                ax.legend(fontsize=8)
            
            # 设置标题和标签
            ax.set_title(f'{features[idx]}', fontsize=10, fontweight='semibold', color='#34495E')
            ax.set_xlabel('feature value', fontsize=10, fontweight='semibold', color='#34495E')
            ax.set_ylabel('pdp', fontsize=10, fontweight='semibold', color='#34495E')
            
            # 添加网格
            ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5, color='#BDC3C7')
            
            # 设置刻度标签大小和颜色
            ax.tick_params(axis='both', which='major', labelsize=9, colors='#2C3E50')
            
            # 设置坐标轴颜色
            ax.spines['bottom'].set_color('#34495E')
            ax.spines['left'].set_color('#34495E')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # 添加统计信息文本
            stats = data['feature_stats']
            stats_text = f"mean: {stats['mean']:.3f}\nmedian: {stats['median']:.3f}"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   fontsize=7, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 隐藏多余的子图
        for idx in range(n_features, len(axes)):
            axes[idx].set_visible(False)
        
        # 调整布局
        plt.tight_layout(pad=2.0)
        
        # 转换为base64字符串
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64


def load_pdp_analyzer(model_path: str, data_path: str, feature_names: List[str]) -> PartialDependenceAnalyzer:
    """
    加载部分依赖图分析器
    
    Args:
        model_path: 模型文件路径
        data_path: 数据文件路径
        feature_names: 特征名称列表
        
    Returns:
        PartialDependenceAnalyzer实例
    """
    # 加载模型
    model = joblib.load(model_path)
    
    # 加载数据
    train_data = pd.read_csv(data_path)
    
    return PartialDependenceAnalyzer(model, feature_names, train_data)