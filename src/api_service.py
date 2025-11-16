"""
基于FastAPI的机器学习模型预测服务
"""

import os
import random
import pandas as pd
import numpy as np
import joblib
import shap
import time
import json
import asyncio
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Body, Response, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from datetime import datetime

# 导入大模型配置
from .llm_config import create_llm

# 导入部分依赖图分析模块
from .partial_dependence import PartialDependenceAnalyzer

# 定义生命周期管理器
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理器"""
    # 应用启动时加载模型和数据（仅在启动时加载，有效避免每次请求都重新加载）
    success = load_model_and_data()
    if not success:
        print("模型和数据加载失败，应用将退出")
        raise RuntimeError("模型和数据加载失败")
    
    # 挂载静态文件目录
    static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static')
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
    
    yield
    # 应用关闭时的清理操作（如果需要的话）
    # 这里可以添加清理代码

# 创建FastAPI应用
app = FastAPI(
    title="机器学习模型预测服务",
    description="使用机器学习模型进行预测并提供SHAP解释的API服务",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件，允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def metrics_middleware(request, call_next):
    """中间件：在每个 HTTP 请求被处理之前和之后执行，用于收集请求指标"""
    start_time = time.time()
    ACTIVE_REQUESTS.inc()
    
    response = await call_next(request)
    
    # 记录请求指标
    request_time = time.time() - start_time
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        http_status=response.status_code
    ).inc()
    
    REQUEST_DURATION.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(request_time)
    
    ACTIVE_REQUESTS.dec()
    
    return response

# 定义Prometheus监控指标
REQUEST_COUNT = Counter(
    'fastapi_requests_total', 
    'Total FastAPI Requests', 
    ['method', 'endpoint', 'http_status']
)

REQUEST_DURATION = Histogram(
    'fastapi_request_duration_seconds', 
    'FastAPI Request Duration',
    ['method', 'endpoint']
)

PREDICTION_COUNT = Counter(
    'model_predictions_total',
    'Total Model Predictions',
    ['prediction_result']
)

ACTIVE_REQUESTS = Gauge(
    'fastapi_active_requests',
    'Active FastAPI Requests'
)

LLM_INTERPRETATION_DURATION = Histogram(
    'llm_interpretation_duration_seconds',
    'LLM Interpretation Duration',
    ['model_type']
)

PREDICTION_OPERATION_DURATION = Histogram(
    'prediction_operation_duration_seconds',
    'Prediction Operation Duration'
)

PREDICTION_PROBABILITY_DISTRIBUTION = Counter(
    'prediction_probability_distribution_total',
    'Prediction Probability Distribution',
    ['prediction_class', 'probability_range']
)

PREDICTION_CLASS_DISTRIBUTION = Counter(
    'prediction_class_distribution_total',
    'Prediction Class Distribution',
    ['prediction_class']
)

# 定义数据模型
class PredictionRequest(BaseModel):
    """请求数据模型"""
    sample_count: Optional[int] = Field(default=1, example=3)
    features: Optional[List[Dict[str, float]]] = Field(
        default=None, 
        example=[
            {
                " Net Value Growth Rate_x_ Equity to Liability": 0.000103828085865,
                " Interest-bearing debt interest rate_div_ Cash/Total Assets": 0.0,
                " Net Value Growth Rate_x_ Revenue Per Share (Yuan ¥)": 1.0,
                " Net Value Growth Rate_x_ Interest-bearing debt interest rate": 0.0,
                " Net profit before tax/Paid-in capital_div_ Interest-bearing debt interest rate": 18269361.284971,
                " Net profit before tax/Paid-in capital_div_ Cash/Total Assets": 0.4278195451575792,
                " Interest-bearing debt interest rate_div_ Revenue Per Share (Yuan ¥)": 0.0,
                " Revenue Per Share (Yuan ¥)_div_ Net Value Growth Rate": 20.98159187831598,
                " Interest-bearing debt interest rate_x_ Net profit before tax/Paid-in capital": 0.0,
                " Net Value Growth Rate_div_ Revenue Per Share (Yuan ¥)": 0.0476597619871344,
                " Net Value Growth Rate_x_ Net profit before tax/Paid-in capital": 8.5732112153338129e-05,
                " Net profit before tax/Paid-in capital_div_ Net Value Growth Rate": 389.308557737556,
                " Net profit before tax/Paid-in capital": 0.18269361284971,
                " Cash/Total Assets": 0.427034273303766,
                " Revenue Per Share (Yuan ¥)_div_ Cash/Total Assets": 0.0230571224692016,
                " Equity to Liability": 0.221255812384907,
                " Cash/Total Assets_div_ Net profit before tax/Paid-in capital": 2.3374339325191387
            }
        ]
    )

class PredictionResponse(BaseModel):
    """响应数据模型"""
    predictions: List[int]
    probabilities: List[float]

class ShapAnalysisRequest(BaseModel):
    """SHAP分析请求数据模型"""
    sample_count: Optional[int] = Field(default=1, example=3)
    features: Optional[List[Dict[str, float]]] = Field(
        default=None, 
        example=[
            {
                " Net Value Growth Rate_x_ Equity to Liability": 0.000103828085865,
                " Interest-bearing debt interest rate_div_ Cash/Total Assets": 0.0,
                " Net Value Growth Rate_x_ Revenue Per Share (Yuan ¥)": 1.0,
                " Net Value Growth Rate_x_ Interest-bearing debt interest rate": 0.0,
                " Net profit before tax/Paid-in capital_div_ Interest-bearing debt interest rate": 18269361.284971,
                " Net profit before tax/Paid-in capital_div_ Cash/Total Assets": 0.4278195451575792,
                " Interest-bearing debt interest rate_div_ Revenue Per Share (Yuan ¥)": 0.0,
                " Revenue Per Share (Yuan ¥)_div_ Net Value Growth Rate": 20.98159187831598,
                " Interest-bearing debt interest rate_x_ Net profit before tax/Paid-in capital": 0.0,
                " Net Value Growth Rate_div_ Revenue Per Share (Yuan ¥)": 0.0476597619871344,
                " Net Value Growth Rate_x_ Net profit before tax/Paid-in capital": 8.5732112153338129e-05,
                " Net profit before tax/Paid-in capital_div_ Net Value Growth Rate": 389.308557737556,
                " Net profit before tax/Paid-in capital": 0.18269361284971,
                " Cash/Total Assets": 0.427034273303766,
                " Revenue Per Share (Yuan ¥)_div_ Cash/Total Assets": 0.0230571224692016,
                " Equity to Liability": 0.221255812384907,
                " Cash/Total Assets_div_ Net profit before tax/Paid-in capital": 2.3374339325191387
            }
        ]
    )
    model: Optional[str] = Field(default="gemini", example="gemini", description="选择的LLM模型: gemini 或 qwen")

class ShapAnalysisResponse(BaseModel):
    """SHAP分析响应数据模型"""
    shap_values: List[List[float]]
    feature_names: List[str]
    base_value: float

class ShapWithStatsResponse(BaseModel):
    """SHAP分析与统计量整合响应数据模型"""
    shap_values: List[List[float]]
    feature_names: List[str]
    base_value: float
    stats_bankrupt_0: Dict[str, Dict[str, float]]
    stats_bankrupt_1: Dict[str, Dict[str, float]]
    llm_interpretation: str

class HealthResponse(BaseModel):
    """健康检查响应模型"""
    status: str
    message: str

class PDPRequest(BaseModel):
    """部分依赖图请求模型"""
    features: Optional[List[str]] = Field(
        default=None, 
        example=["Net profit before tax/Paid-in capital", "Cash/Total Assets"],
        description="要分析的特征列表，最多9个特征。如果为空，将自动选择重要特征"
    )
    sample_data: Optional[Dict[str, float]] = Field(
        default=None,
        description="可选的样本数据，用于在PDP图中标注样本点"
    )
    grid_resolution: Optional[int] = Field(
        default=100,
        ge=20,
        le=200,
        description="网格分辨率，控制PDP曲线的平滑度"
    )
    use_multiprocessing: Optional[bool] = Field(
        default=False,
        description="是否使用多进程计算PDP"
    )
    n_processes: Optional[int] = Field(
        default=None,
        ge=1,
        le=16,
        description="进程数，如果为None则使用CPU核心数"
    )

class PDPResponse(BaseModel):
    """PDP分析响应数据模型"""
    pdp_data: Dict[str, Any] = Field(description="部分依赖图数据")
    plot_image: str = Field(description="Base64编码的PDP图像")
    selected_features: List[str] = Field(description="实际分析的特征列表")
    calculation_time: float = Field(description="计算时间（秒）")
    use_multiprocessing: bool = Field(description="是否使用了多进程")
    n_processes: Optional[int] = Field(description="使用的进程数")

class PDPDataResponse(BaseModel):
    """PDP数据响应模型"""
    pdp_data: Dict[str, Any] = Field(description="部分依赖图数据")
    selected_features: List[str] = Field(description="实际分析的特征列表")
    calculation_time: float = Field(description="计算时间（秒）")
    use_multiprocessing: bool = Field(description="是否使用了多进程")
    n_processes: Optional[int] = Field(description="使用的进程数")

class ModelInfo(BaseModel):
    """模型信息模型"""
    key: str = Field(description="模型键名")
    name: str = Field(description="模型显示名称")
    description: str = Field(description="模型描述")
    file: str = Field(description="模型文件名")
    is_available: bool = Field(description="模型是否可用")
    is_current: bool = Field(description="是否为当前使用的模型")

class ModelsResponse(BaseModel):
    """可用模型列表响应模型"""
    models: List[ModelInfo] = Field(description="可用模型列表")
    current_model: str = Field(description="当前使用的模型键名")

class SwitchModelResponse(BaseModel):
    """切换模型响应模型"""
    message: str = Field(description="操作结果消息")
    current_model: str = Field(description="当前模型键名")
    previous_model: Optional[str] = Field(description="之前的模型键名")

class CurrentModelResponse(BaseModel):
    """当前模型信息响应模型"""
    key: str = Field(description="模型键名")
    name: str = Field(description="模型显示名称")
    description: str = Field(description="模型描述")
    file: str = Field(description="模型文件名")
    feature_count: int = Field(description="特征数量")
    test_data_count: int = Field(description="测试数据条数")
    train_data_count: int = Field(description="训练数据条数")

def get_probability_range(probability):
    """
    根据概率值确定概率范围
    
    参数:
    - probability: 概率值 (0.0-1.0)
    
    返回:
    - 概率范围字符串
    """
    if probability < 0.1:
        return "0.0-0.1"
    elif probability < 0.2:
        return "0.1-0.2"
    elif probability < 0.3:
        return "0.2-0.3"
    elif probability < 0.4:
        return "0.3-0.4"
    elif probability < 0.5:
        return "0.4-0.5"
    elif probability < 0.6:
        return "0.5-0.6"
    elif probability < 0.7:
        return "0.6-0.7"
    elif probability < 0.8:
        return "0.7-0.8"
    elif probability < 0.9:
        return "0.8-0.9"
    else:
        return "0.9-1.0"

# 全局变量，用于存储模型和解释器
model = None
explainer = None
feature_names = None
test_data = None
train_data = None
pdp_analyzer = None
current_model_name = "xgboost_binned_model"  # 默认模型

# 模型配置映射
MODEL_CONFIG = {
    "logistic_model": {
        "file": "logistic_model.joblib",
        "name": "逻辑回归模型",
        "description": "基于逻辑回归算法的二分类模型，适用于线性可分的数据"
    },
    "rf_model": {
        "file": "rf_model.joblib", 
        "name": "随机森林模型",
        "description": "基于随机森林算法的集成学习模型，具有良好的泛化能力"
    },
    "xgb_model": {
        "file": "xgb_model.joblib",
        "name": "XGBoost模型",
        "description": "基于梯度提升决策树的高性能机器学习模型"
    },
    "xgboost_binned_model": {
        "file": "xgboost_binned_model.joblib",
        "name": "XGBoost分箱模型", 
        "description": "使用分箱特征工程的XGBoost模型，提高了模型的稳定性和解释性"
    }
}

def get_data_files_for_model(model_name):
    """获取binned数据文件路径（所有模型统一使用binned数据）"""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    test_data_path = os.path.join(base_dir, 'data', 'binned_test_data.csv')
    train_data_path = os.path.join(base_dir, 'data', 'binned_train_data.csv')
    
    return test_data_path, train_data_path

def load_model_and_data(model_name=None):
    """加载指定模型和对应数据"""
    global model, explainer, feature_names, test_data, train_data, pdp_analyzer, current_model_name
    
    if model_name is None:
        model_name = current_model_name
    
    try:
        # 检查模型是否存在于配置中
        if model_name not in MODEL_CONFIG:
            print(f"错误: 未知的模型名称: {model_name}")
            return False
        
        config = MODEL_CONFIG[model_name]
        
        # 模型文件路径
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', config["file"])
        
        # 获取对应的数据文件路径
        test_data_path, train_data_path = get_data_files_for_model(model_name)
        
        # 检查文件是否存在
        if not os.path.exists(model_path):
            print(f"错误: 模型文件不存在: {model_path}")
            return False
        if not os.path.exists(test_data_path):
            print(f"错误: 测试数据文件不存在: {test_data_path}")
            return False
        if not os.path.exists(train_data_path):
            print(f"错误: 训练数据文件不存在: {train_data_path}")
            return False
        
        # 加载模型
        print(f"正在加载模型: {config['name']} ({config['file']})...")
        model = joblib.load(model_path)
        current_model_name = model_name
        print("模型加载成功")
        
        # 加载测试数据
        print("正在加载测试数据...")
        test_data = pd.read_csv(test_data_path)
        print(f"测试数据加载成功，共 {len(test_data)} 条记录")
        
        # 加载训练数据
        print("正在加载训练数据...")
        train_data = pd.read_csv(train_data_path)
        print(f"训练数据加载成功，共 {len(train_data)} 条记录")
        
        # 获取特征名称（除了最后一列是目标变量）
        feature_names = test_data.columns[:-1].tolist()
        print(f"特征数量: {len(feature_names)}")
        
        # 创建SHAP解释器
        print("正在创建SHAP解释器...")
        model_type = type(model).__name__
        print(f"模型类型: {model_type}")
        
        if 'LogisticRegression' in model_type:
            # 逻辑回归模型使用LinearExplainer
            explainer = shap.LinearExplainer(model, train_data.iloc[:, :-1])
            print("使用LinearExplainer创建SHAP解释器")
        elif any(tree_type in model_type for tree_type in ['XGB', 'RandomForest', 'GradientBoosting', 'DecisionTree']):
            # 基于树的模型使用TreeExplainer
            explainer = shap.TreeExplainer(model)
            print("使用TreeExplainer创建SHAP解释器")
        else:
            # 其他模型使用KernelExplainer（通用但较慢）
            background_data = shap.sample(train_data.iloc[:, :-1], 100)
            explainer = shap.KernelExplainer(model.predict_proba, background_data)
            print("使用KernelExplainer创建SHAP解释器")
        
        print("SHAP解释器创建成功")
        
        # 测试模型预测功能
        print("测试模型预测功能...")
        sample_data = test_data.iloc[:1, :-1]
        prediction = model.predict(sample_data)
        probability = model.predict_proba(sample_data)
        print(f"测试预测结果: 类别={prediction[0]}, 概率={probability[0]}")
        
        # 测试SHAP计算功能
        print("测试SHAP计算功能...")
        shap_values = explainer.shap_values(sample_data)
        print(f"SHAP值计算成功，形状: {np.array(shap_values).shape}")
        
        # 创建部分依赖图分析器
        print("正在创建部分依赖图分析器...")
        pdp_analyzer = PartialDependenceAnalyzer(model, feature_names, train_data)
        print("部分依赖图分析器创建成功")
        
        print(f"模型 {config['name']} 和数据加载成功")
        return True
    except Exception as e:
        print(f"加载模型和数据时出错: {e}")
        import traceback
        traceback.print_exc()
        return False


@app.get("/")
async def root():
    """根路径，返回测试页面"""
    static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static')
    index_path = os.path.join(static_dir, 'index.html')
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "机器学习模型预测服务运行中", "docs_url": "/docs"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查端点"""
    if model is None:
        raise HTTPException(status_code=500, detail="模型未加载")
    return HealthResponse(status="healthy", message="模型预测服务运行正常")

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest = Body(
        examples={
            "sample_count_example": {
                "summary": "使用样本数量",
                "description": "使用样本数量进行预测",
                "value": {
                    "sample_count": 3,
                    "features": None
                }
            },
            "features_example": {
                "summary": "使用特征数据",
                "description": "直接提供特征数据进行预测",
                "value": {
                    "sample_count": 1,
                    "features": [
                        {
                            " Net Value Growth Rate_x_ Equity to Liability": 0.000103828085865,
                            " Interest-bearing debt interest rate_div_ Cash/Total Assets": 0.0,
                            " Net Value Growth Rate_x_ Revenue Per Share (Yuan ¥)": 1.0,
                            " Net Value Growth Rate_x_ Interest-bearing debt interest rate": 0.0,
                            " Net profit before tax/Paid-in capital_div_ Interest-bearing debt interest rate": 18269361.284971,
                            " Net profit before tax/Paid-in capital_div_ Cash/Total Assets": 0.4278195451575792,
                            " Interest-bearing debt interest rate_div_ Revenue Per Share (Yuan ¥)": 0.0,
                            " Revenue Per Share (Yuan ¥)_div_ Net Value Growth Rate": 20.98159187831598,
                            " Interest-bearing debt interest rate_x_ Net profit before tax/Paid-in capital": 0.0,
                            " Net Value Growth Rate_div_ Revenue Per Share (Yuan ¥)": 0.0476597619871344,
                            " Net Value Growth Rate_x_ Net profit before tax/Paid-in capital": 8.5732112153338129e-05,
                            " Net profit before tax/Paid-in capital_div_ Net Value Growth Rate": 389.308557737556,
                            " Net profit before tax/Paid-in capital": 0.18269361284971,
                            " Cash/Total Assets": 0.427034273303766,
                            " Revenue Per Share (Yuan ¥)_div_ Cash/Total Assets": 0.0230571224692016,
                            " Equity to Liability": 0.221255812384907,
                            " Cash/Total Assets_div_ Net profit before tax/Paid-in capital": 2.3374339325191387
                        }
                    ]
                }
            }
        }
    )
):
    """
    预测端点
    
    参数:
    - request: 预测请求，可以指定样本数量或直接提供特征数据
    
    返回:
    - 预测结果和概率
    """
    if model is None or test_data is None:
        raise HTTPException(status_code=500, detail="模型或数据未加载")
    
    try:
        print(f"预测请求: sample_count={request.sample_count}, features={request.features is not None}")
        
        # 如果直接提供了特征数据
        if request.features is not None and request.features:
            print(f"提供的特征数据: {len(request.features)} 条记录")
            print(f"第一条记录的特征: {list(request.features[0].keys())}")
            
            # 将提供的特征数据转换为DataFrame
            input_data = pd.DataFrame(request.features)
            
            # 确保所有特征都存在
            missing_features = set(feature_names) - set(input_data.columns)
            if missing_features:
                error_msg = f"缺少特征: {missing_features}。请确保提供所有特征。"
                print(f"错误: {error_msg}")
                raise HTTPException(status_code=400, detail=error_msg)
            
            # 确保特征顺序正确
            input_data = input_data[feature_names]
        else:
            # 从测试数据中随机抽取样本
            sample_count = request.sample_count if request.sample_count > 0 else 1
            sample_count = min(sample_count, len(test_data))
            
            # 随机抽取样本
            sample_indices = random.sample(range(len(test_data)), sample_count)
            input_data = test_data.iloc[sample_indices, :-1]  # 排除目标变量
            print(f"随机抽取 {sample_count} 个样本进行预测")
        
        print(f"输入数据形状: {input_data.shape}")
        
        # 记录预测操作开始时间
        prediction_start_time = time.time()
        
        # 进行预测
        predictions = model.predict(input_data)
        probabilities = model.predict_proba(input_data)[:, 1]
        
        # 记录预测操作耗时
        prediction_duration = time.time() - prediction_start_time
        PREDICTION_OPERATION_DURATION.observe(prediction_duration)
        
        print(f"预测完成: 类别={predictions.tolist()}, 概率={probabilities.tolist()}")

        # 记录预测指标
        for pred, prob in zip(predictions.tolist(), probabilities.tolist()):
            PREDICTION_COUNT.labels(prediction_result=str(pred)).inc()
            
            # 记录预测结果的类别分布
            PREDICTION_CLASS_DISTRIBUTION.labels(prediction_class=str(pred)).inc()
            
            # 记录预测结果的概率分布
            # 根据概率值确定概率范围
            prob_range = get_probability_range(prob)
            PREDICTION_PROBABILITY_DISTRIBUTION.labels(prediction_class=str(pred), probability_range=prob_range).inc()

        return PredictionResponse(
            predictions=predictions.tolist(),
            probabilities=probabilities.tolist()
        )
    except HTTPException:
        # 直接传递HTTP异常
        raise
    except Exception as e:
        error_msg = f"预测过程中出错: {str(e)}"
        print(f"错误: {error_msg}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/shap-analysis", response_model=ShapAnalysisResponse)
async def shap_analysis(
    request: ShapAnalysisRequest = Body(
        examples={
            "sample_count_example": {
                "summary": "使用样本数量",
                "description": "使用样本数量进行SHAP分析",
                "value": {
                    "sample_count": 3,
                    "features": None
                }
            },
            "features_example": {
                "summary": "使用特征数据",
                "description": "直接提供特征数据进行SHAP分析",
                "value": {
                    "sample_count": 1,
                    "features": [
                        {
                            " Net Value Growth Rate_x_ Equity to Liability": 0.000103828085865,
                            " Interest-bearing debt interest rate_div_ Cash/Total Assets": 0.0,
                            " Net Value Growth Rate_x_ Revenue Per Share (Yuan ¥)": 1.0,
                            " Net Value Growth Rate_x_ Interest-bearing debt interest rate": 0.0,
                            " Net profit before tax/Paid-in capital_div_ Interest-bearing debt interest rate": 18269361.284971,
                            " Net profit before tax/Paid-in capital_div_ Cash/Total Assets": 0.4278195451575792,
                            " Interest-bearing debt interest rate_div_ Revenue Per Share (Yuan ¥)": 0.0,
                            " Revenue Per Share (Yuan ¥)_div_ Net Value Growth Rate": 20.98159187831598,
                            " Interest-bearing debt interest rate_x_ Net profit before tax/Paid-in capital": 0.0,
                            " Net Value Growth Rate_div_ Revenue Per Share (Yuan ¥)": 0.0476597619871344,
                            " Net Value Growth Rate_x_ Net profit before tax/Paid-in capital": 8.5732112153338129e-05,
                            " Net profit before tax/Paid-in capital_div_ Net Value Growth Rate": 389.308557737556,
                            " Net profit before tax/Paid-in capital": 0.18269361284971,
                            " Cash/Total Assets": 0.427034273303766,
                            " Revenue Per Share (Yuan ¥)_div_ Cash/Total Assets": 0.0230571224692016,
                            " Equity to Liability": 0.221255812384907,
                            " Cash/Total Assets_div_ Net profit before tax/Paid-in capital": 2.3374339325191387
                        }
                    ]
                }
            }
        }
    )
):
    """
    SHAP分析端点
    
    参数:
    - request: SHAP分析请求，可以指定样本数量或直接提供特征数据
    
    返回:
    - SHAP解释值、特征名称和基准值
    """
    if model is None or explainer is None or test_data is None:
        raise HTTPException(status_code=500, detail="模型或数据未加载")
    
    try:
        print(f"SHAP分析请求: sample_count={request.sample_count}, features={request.features is not None}")
        
        # 如果直接提供了特征数据
        if request.features is not None and request.features:
            print(f"提供的特征数据: {len(request.features)} 条记录")
            print(f"第一条记录的特征: {list(request.features[0].keys())}")
            
            # 将提供的特征数据转换为DataFrame
            input_data = pd.DataFrame(request.features)
            
            # 确保所有特征都存在
            missing_features = set(feature_names) - set(input_data.columns)
            if missing_features:
                error_msg = f"缺少特征: {missing_features}。请确保提供所有17个特征。"
                print(f"错误: {error_msg}")
                raise HTTPException(status_code=400, detail=error_msg)
            
            # 确保特征顺序正确
            input_data = input_data[feature_names]
        else:
            # 从测试数据中随机抽取样本
            sample_count = request.sample_count if request.sample_count > 0 else 1
            sample_count = min(sample_count, len(test_data))
            
            # 随机抽取样本
            sample_indices = random.sample(range(len(test_data)), sample_count)
            input_data = test_data.iloc[sample_indices, :-1]  # 排除目标变量
            print(f"随机抽取 {sample_count} 个样本进行SHAP分析")
        
        print(f"输入数据形状: {input_data.shape}")
        
        # 计算SHAP值
        shap_values = explainer.shap_values(input_data)
        
        # 获取基准值
        base_value = explainer.expected_value
        
        # 如果模型是多输出的，取第一个输出的基准值
        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[0]
        
        # 将SHAP值转换为列表格式
        shap_values_list = shap_values.tolist()
        
        print(f"SHAP分析完成: 基准值={base_value}, SHAP值形状={np.array(shap_values).shape}")
        
        return ShapAnalysisResponse(
            shap_values=shap_values_list,
            feature_names=feature_names,
            base_value=float(base_value)
        )
    except HTTPException:
        # 直接传递HTTP异常
        raise
    except Exception as e:
        error_msg = f"SHAP分析过程中出错: {str(e)}"
        print(f"错误: {error_msg}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/features")
async def get_features():
    """获取特征名称列表"""
    if feature_names is None:
        raise HTTPException(status_code=500, detail="特征数据未加载")
    return {"features": feature_names}

@app.get("/sample")
async def get_sample(count: int = 1):
    """
    获取随机样本数据
    
    参数:
    - count: 要获取的样本数量
    
    返回:
    - 随机样本数据
    """
    if test_data is None:
        raise HTTPException(status_code=500, detail="测试数据未加载")
    
    count = max(1, min(count, 10))  # 限制最多返回10个样本
    
    # 随机抽取样本
    sample_indices = random.sample(range(len(test_data)), count)
    samples = test_data.iloc[sample_indices].to_dict('records')
    
    return {"samples": samples}

@app.post("/pdp-analysis", response_model=PDPResponse)
async def pdp_analysis(request: PDPRequest):
    """
    生成部分依赖图分析
    
    参数:
    - features: 要分析的特征列表（最多9个）
    - sample_data: 可选的样本数据，用于在图中标注
    - grid_resolution: 网格分辨率
    
    返回:
    - PDP数据和Base64编码的图像
    """
    if pdp_analyzer is None:
        raise HTTPException(status_code=500, detail="PDP分析器未初始化")
    
    try:
        # 获取要分析的特征
        features_to_analyze = request.features
        
        # 验证特征是否存在
        invalid_features = [f for f in features_to_analyze if f not in feature_names]
        if invalid_features:
            raise HTTPException(
                status_code=400, 
                detail=f"无效的特征名称: {invalid_features}"
            )
        
        # 计算PDP数据
        pdp_data, calculation_time = pdp_analyzer.calculate_partial_dependence(
            features=features_to_analyze,
            grid_resolution=request.grid_resolution,
            use_multiprocessing=request.use_multiprocessing,
            n_processes=request.n_processes
        )
        
        # 生成PDP图像（使用已计算的数据，避免重复计算）
        plot_image, plot_calculation_time = pdp_analyzer.create_pdp_plots(
            features=features_to_analyze,
            sample_data=request.sample_data,
            grid_resolution=request.grid_resolution,
            pdp_data=pdp_data,
            use_multiprocessing=request.use_multiprocessing,
            n_processes=request.n_processes
        )
        
        return PDPResponse(
            pdp_data=pdp_data,
            plot_image=plot_image,
            selected_features=features_to_analyze,
            calculation_time=calculation_time,
            use_multiprocessing=request.use_multiprocessing or False,
            n_processes=request.n_processes
        )
        
    except Exception as e:
        error_msg = f"PDP分析失败: {str(e)}"
        print(f"Error in PDP analysis: {error_msg}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/pdp-data", response_model=PDPDataResponse)
async def pdp_data(request: PDPRequest):
    """
    获取部分依赖图数据（不生成图像）
    
    参数:
    - features: 要分析的特征列表（最多9个）
    - grid_resolution: 网格分辨率
    
    返回:
    - PDP数据
    """
    if pdp_analyzer is None:
        raise HTTPException(status_code=500, detail="PDP分析器未初始化")
    
    try:
        # 获取要分析的特征
        features_to_analyze = request.features
        if features_to_analyze is None:
            # 如果没有指定特征，使用特征重要性选择前9个
            feature_importance = pdp_analyzer.get_feature_importance()
            features_to_analyze = list(feature_importance.keys())[:9]
        else:
            # 限制最多9个特征
            features_to_analyze = features_to_analyze[:9]
        
        # 验证特征是否存在
        invalid_features = [f for f in features_to_analyze if f not in feature_names]
        if invalid_features:
            raise HTTPException(
                status_code=400, 
                detail=f"无效的特征名称: {invalid_features}"
            )
        
        # 计算PDP数据
        pdp_data, calculation_time = pdp_analyzer.calculate_partial_dependence(
            features=features_to_analyze,
            grid_resolution=request.grid_resolution,
            use_multiprocessing=request.use_multiprocessing,
            n_processes=request.n_processes
        )
        
        return PDPDataResponse(
            pdp_data=pdp_data,
            selected_features=features_to_analyze,
            calculation_time=calculation_time,
            use_multiprocessing=request.use_multiprocessing or False,
            n_processes=request.n_processes
        )
        
    except Exception as e:
        error_msg = f"PDP数据获取失败: {str(e)}"
        print(f"Error in PDP data: {error_msg}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)

def calculate_feature_stats(data, feature_names):
    """
    计算特征的统计量信息
    
    参数:
    - data: 数据DataFrame
    - feature_names: 特征名称列表
    
    返回:
    - 特征统计量字典
    """
    stats = {}
    for feature in feature_names:
        if feature in data.columns:
            stats[feature] = {
                'mean': float(data[feature].mean()),
                'std': float(data[feature].std()),
                'min': float(data[feature].min()),
                'max': float(data[feature].max()),
                'median': float(data[feature].median()),
                'q25': float(data[feature].quantile(0.25)),
                'q75': float(data[feature].quantile(0.75))
            }
    return stats

def format_stats_for_llm(stats_bankrupt_0, stats_bankrupt_1, feature_names, top_features=None):
    """
    格式化统计量信息，便于LLM解读

    参数:
    - stats_bankrupt_0: Bankrupt=0的统计量
    - stats_bankrupt_1: Bankrupt=1的统计量
    - feature_names: 特征名称列表
    - top_features: 需要显示的特征列表，如果为None则显示所有特征

    返回:
    - 格式化后的统计量文本
    """
    # 获取Bankrupt=0和Bankrupt=1的样本数
    # 为了简化，这里直接从stats_bankrupt_0/1中获取任意一个特征的统计量来推断样本数
    sample_count_0 = len(train_data[train_data['Bankrupt']==0])
    sample_count_1 = len(train_data[train_data['Bankrupt']==1])

    formatted_text = f"特征统计量信息（按Bankrupt标签分组）：\n"
    formatted_text += f"  Bankrupt=0 样本总数: {sample_count_0}\n"
    formatted_text += f"  Bankrupt=1 样本总数: {sample_count_1}\n\n"

    # 确定要显示的特征
    features_to_show = top_features if top_features is not None else feature_names

    for feature in features_to_show:
        if feature in stats_bankrupt_0 and feature in stats_bankrupt_1:
            stats_0 = stats_bankrupt_0[feature]
            stats_1 = stats_bankrupt_1[feature]

            formatted_text += f"特征: {feature}\n"
            formatted_text += f"  Bankrupt=0:\n"
            formatted_text += f"    均值: {stats_0['mean']:.4f}, 标准差: {stats_0['std']:.4f}\n"
            formatted_text += f"    最小值: {stats_0['min']:.4f}, 最大值: {stats_0['max']:.4f}\n"
            formatted_text += f"    中位数: {stats_0['median']:.4f}\n"

            formatted_text += f"  Bankrupt=1:\n"
            formatted_text += f"    均值: {stats_1['mean']:.4f}, 标准差: {stats_1['std']:.4f}\n"
            formatted_text += f"    最小值: {stats_1['min']:.4f}, 最大值: {stats_1['max']:.4f}\n"
            formatted_text += f"    中位数: {stats_1['median']:.4f}\n"

    return formatted_text


def format_shap_for_llm(shap_values, feature_names, input_data):
    """
    格式化SHAP值信息，便于LLM解读
    
    参数:
    - shap_values: SHAP值数组
    - feature_names: 特征名称列表
    - input_data: 输入数据
    
    返回:
    - 格式化后的SHAP值文本
    """
    formatted_text = "SHAP值分析结果：\n\n"
    
    for i, sample_shap_values in enumerate(shap_values):
        formatted_text += f"样本 {i+1}:\n"
        
        # 创建特征名、SHAP值和特征值的元组列表
        feature_shap_value_tuples = []
        for j, (feature, shap_val) in enumerate(zip(feature_names, sample_shap_values)):
            feature_val = input_data.iloc[i, j] if hasattr(input_data, 'iloc') else None
            feature_shap_value_tuples.append((feature, shap_val, feature_val))
        
        # 按SHAP值绝对值降序排序
        feature_shap_value_tuples.sort(key=lambda x: abs(x[1]), reverse=True)
        
        formatted_text += "  特征重要性排序（按SHAP值绝对值）:\n"
        for feature, shap_val, feature_val in feature_shap_value_tuples[:5]:  # 只显示前5个最重要的特征
            direction = "正向" if shap_val > 0 else "负向"
            formatted_text += f"    {feature}: SHAP值={shap_val:.4f} ({direction}影响)"
            if feature_val is not None:
                formatted_text += f", 特征值={feature_val:.4f}"
            formatted_text += "\n"
        
        formatted_text += "\n"
    
    return formatted_text

# 创建一个 实时双向通信 的 WebSocket 端点
@app.websocket("/ws/shap-with-stats-analysis")
async def websocket_shap_with_stats_analysis(websocket: WebSocket):
    """
    WebSocket端点：SHAP分析与统计量整合（流式输出）
    """
    await websocket.accept()
    
    try:
        while True:
            # 接收客户端请求
            data = await websocket.receive_text()  # 接收客户端文本消息
            request_data = json.loads(data)  # 是一个字典，包含从WebSocket接收到的JSON数据
            
            # 验证请求数据格式
            try:
                # 将request_data字典解包后创建ShapAnalysisRequest对象
                request = ShapAnalysisRequest(**request_data)  
            except Exception as e:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"请求数据格式错误: {str(e)}"
                }))
                continue
            
            if model is None or explainer is None or test_data is None or train_data is None:
                # 发送错误信息给客户端
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "模型或数据未加载"
                }))
                continue
            
            try:
                print(f"WebSocket SHAP与统计量分析请求: sample_count={request.sample_count}, features={request.features is not None}")
                
                # 发送开始分析消息
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "message": "开始SHAP分析..."
                }))
                
                # 处理输入数据
                if request.features is not None and request.features:
                    print(f"提供的特征数据: {len(request.features)} 条记录")
                    input_data = pd.DataFrame(request.features)
                    
                    missing_features = set(feature_names) - set(input_data.columns)
                    if missing_features:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": f"缺少特征: {missing_features}。请确保提供所有17个特征。"
                        }))
                        continue
                    
                    input_data = input_data[feature_names]
                else:
                    sample_count = request.sample_count if request.sample_count > 0 else 1
                    sample_count = min(sample_count, len(test_data))
                    sample_indices = random.sample(range(len(test_data)), sample_count)
                    input_data = test_data.iloc[sample_indices, :-1]
                    print(f"随机抽取 {sample_count} 个样本进行SHAP与统计量分析")
                
                print(f"输入数据形状: {input_data.shape}")
                
                # 计算SHAP值
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "message": "计算SHAP值..."
                }))
                
                # 计算SHAP值
                shap_values = explainer.shap_values(input_data)
                base_value = explainer.expected_value
                
                if isinstance(base_value, (list, np.ndarray)):
                    base_value = base_value[0]
                
                shap_values_list = shap_values.tolist()
                
                print(f"SHAP分析完成: 基准值={base_value}, SHAP值形状={np.array(shap_values).shape}")
                
                # 计算分组统计量
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "message": "计算分组统计量..."
                }))
                
                bankrupt_0_data = train_data[train_data['Bankrupt'] == 0]
                bankrupt_1_data = train_data[train_data['Bankrupt'] == 1]
                
                # 使用异步执行避免阻塞事件循环
                stats_bankrupt_0 = await asyncio.to_thread(calculate_feature_stats, bankrupt_0_data, feature_names)
                stats_bankrupt_1 = await asyncio.to_thread(calculate_feature_stats, bankrupt_1_data, feature_names)
                
                print(f"分组统计量计算完成: Bankrupt=0有{len(bankrupt_0_data)}条样本, Bankrupt=1有{len(bankrupt_1_data)}条样本")
                
                # 发送SHAP和统计量结果
                await websocket.send_text(json.dumps({
                    "type": "shap_results",
                    "data": {
                        "shap_values": shap_values_list,
                        "feature_names": feature_names,
                        "base_value": float(base_value),
                        "stats_bankrupt_0": stats_bankrupt_0,
                        "stats_bankrupt_1": stats_bankrupt_1
                    }
                }))
                
                # 发送数据处理状态消息
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "message": "正在处理SHAP结果和统计数据..."
                }))
                
                # 获取前5个最重要的特征
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "message": "正在分析重要特征..."
                }))
                
                top_features = []
                for i, sample_shap_values in enumerate(shap_values):
                    feature_shap_value_tuples = []
                    for j, (feature, shap_val) in enumerate(zip(feature_names, sample_shap_values)):
                        feature_val = input_data.iloc[i, j] if hasattr(input_data, 'iloc') else None
                        feature_shap_value_tuples.append((feature, shap_val, feature_val))
                    
                    feature_shap_value_tuples.sort(key=lambda x: abs(x[1]), reverse=True)
                    sample_top_features = [feature for feature, _, _ in feature_shap_value_tuples[:5]]
                    top_features.extend(sample_top_features)
                
                seen = set()
                top_features = [x for x in top_features if not (x in seen or seen.add(x))]
                
                # 发送数据格式化状态消息
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "message": "正在准备LLM分析数据..."
                }))
                
                # 格式化数据准备提交给LLM
                stats_text = format_stats_for_llm(stats_bankrupt_0, stats_bankrupt_1, feature_names, top_features)
                shap_text = format_shap_for_llm(shap_values, feature_names, input_data)
                print(f"stats_text是：{stats_text}")
                print(f"shap_text是：{shap_text}")

                # 构建LLM提示
                llm_prompt = f"""
你是一位专业的数据科学家和机器学习专家。请分析以下SHAP值和特征统计量信息，并提供专业且清晰易懂的解读。

{stats_text}
{shap_text}

请提供简洁的解读。
"""
                
                # 发送LLM准备状态
                await websocket.send_text(json.dumps({
                    "type": "status",
                    "message": "正在连接LLM模型..."
                }))
                
                # 开始LLM流式解读
                await websocket.send_text(json.dumps({
                    "type": "llm_start",
                    "message": "开始LLM解读..."
                }))
                
                # 获取选择的模型
                selected_model = request.model if hasattr(request, 'model') and request.model else "gemini"
                print(f"正在调用{selected_model}大模型进行流式解读...")
                llm_manager = create_llm(selected_model)
                
                # 记录LLM解读操作开始时间
                llm_start_time = time.time()
                print(f"LLM开始时间: {llm_start_time}")
                
                # 使用流式响应
                try:
                    chunk_count = 0
                    for chunk in llm_manager.get_streaming_response(
                        llm_prompt,
                        system_prompt="你是一位专业的数据科学家和机器学习专家，擅长解释SHAP值和统计分析结果。"
                    ):
                        chunk_count += 1
                        if chunk_count == 1:
                            first_chunk_time = time.time()
                            print(f"首个chunk接收时间: {first_chunk_time}, 延迟: {first_chunk_time - llm_start_time:.2f}秒")
                        
                        # 发送流式内容
                        await websocket.send_text(json.dumps({
                            "type": "llm_chunk",
                            "content": chunk
                        }))
                        await asyncio.sleep(0.01)
                        print(chunk, end='', flush=True)
                        
                except Exception as stream_error:
                    import traceback
                    error_details = traceback.format_exc()
                    print(f"流式响应错误: {stream_error}")
                    print(f"错误详情: {error_details}")
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": f"流式响应错误: {str(stream_error)}"
                    }))
                    return
                
                # 记录LLM解读操作耗时
                llm_duration = time.time() - llm_start_time
                LLM_INTERPRETATION_DURATION.labels(model_type=selected_model).observe(llm_duration)
                
                # 发送完成消息
                await websocket.send_text(json.dumps({
                    "type": "llm_complete",
                    "message": "LLM解读完成"
                }))
                
                print(f"{selected_model}大模型流式解读完成")
            except Exception as e:
                error_msg = f"SHAP与统计量分析过程中出错: {str(e)}"
                print(f"错误: {error_msg}")
                import traceback
                traceback.print_exc()
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": error_msg
                }))
                
    except WebSocketDisconnect:
        print("WebSocket连接断开")
    except Exception as e:
        print(f"WebSocket错误: {str(e)}")

@app.get("/models", response_model=ModelsResponse)
async def get_available_models():
    """获取可用模型列表"""
    models_info = []
    for model_key, config in MODEL_CONFIG.items():
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', config["file"])
        test_data_path, train_data_path = get_data_files_for_model(model_key)
        
        # 检查文件是否存在
        is_available = (
            os.path.exists(model_path) and 
            os.path.exists(test_data_path) and 
            os.path.exists(train_data_path)
        )
        
        models_info.append({
            "key": model_key,
            "name": config["name"],
            "description": config["description"],
            "file": config["file"],
            "is_available": is_available,
            "is_current": model_key == current_model_name
        })
    
    return {
        "models": models_info,
        "current_model": current_model_name
    }

@app.post("/switch-model", response_model=SwitchModelResponse)
async def switch_model(model_key: str = Body(..., embed=True)):
    """切换模型"""
    if model_key not in MODEL_CONFIG:
        raise HTTPException(status_code=400, detail=f"未知的模型名称: {model_key}")
    
    if model_key == current_model_name:
        return {
            "message": f"当前已经是模型 {MODEL_CONFIG[model_key]['name']}",
            "current_model": current_model_name
        }
    
    try:
        success = load_model_and_data(model_key)
        if success:
            return {
                "message": f"成功切换到模型 {MODEL_CONFIG[model_key]['name']}",
                "current_model": current_model_name,
                "previous_model": model_key if model_key != current_model_name else None
            }
        else:
            raise HTTPException(status_code=500, detail=f"模型 {model_key} 加载失败")
    except Exception as e:
        error_msg = f"切换模型时出错: {str(e)}"
        print(f"错误: {error_msg}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/current-model", response_model=CurrentModelResponse)
async def get_current_model():
    """获取当前使用的模型信息"""
    if current_model_name not in MODEL_CONFIG:
        raise HTTPException(status_code=500, detail="当前模型配置无效")
    
    config = MODEL_CONFIG[current_model_name]
    return {
        "key": current_model_name,
        "name": config["name"],
        "description": config["description"],
        "file": config["file"],
        "feature_count": len(feature_names) if feature_names else 0,
        "test_data_count": len(test_data) if test_data is not None else 0,
        "train_data_count": len(train_data) if train_data is not None else 0
    }

@app.get("/metrics")
async def metrics():
    """Prometheus监控指标端点"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)