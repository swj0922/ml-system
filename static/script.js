// API基础URL
const API_BASE = '';

// 页面加载时检查API状态并获取特征列表
document.addEventListener('DOMContentLoaded', function() {
    checkApiStatus();
    getFeatures();
});

// 检查API状态
async function checkApiStatus() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();
        document.getElementById('api-status').textContent = data.status === 'healthy' ? '正常运行' : '异常';
        document.getElementById('api-status').style.color = data.status === 'healthy' ? '#2ecc71' : '#e74c3c';
    } catch (error) {
        document.getElementById('api-status').textContent = '无法连接';
        document.getElementById('api-status').style.color = '#e74c3c';
    }
}

// 获取特征列表
let featureNames = [];
async function getFeatures() {
    try {
        const response = await fetch(`${API_BASE}/features`);
        const data = await response.json();
        featureNames = data.features;
        createFeatureInputs();
        createShapFeatureInputs();
        createShapStatsFeatureInputs();
    } catch (error) {
        showError('获取特征列表失败: ' + error.message);
    }
}

// 创建特征输入表单
function createFeatureInputs() {
    const container = document.getElementById('feature-inputs');
    container.innerHTML = '';
    
    featureNames.forEach(feature => {
        const inputGroup = document.createElement('div');
        inputGroup.className = 'input-group';
        
        const label = document.createElement('label');
        label.textContent = feature;
        
        const input = document.createElement('input');
        input.type = 'number';
        input.step = 'any';
        input.id = `feature-${feature}`;
        input.placeholder = `输入 ${feature} 的值`;
        
        inputGroup.appendChild(label);
        inputGroup.appendChild(input);
        container.appendChild(inputGroup);
    });
}

// 创建SHAP特征输入表单
function createShapFeatureInputs() {
    const container = document.getElementById('shap-feature-inputs');
    container.innerHTML = '';
    
    featureNames.forEach(feature => {
        const inputGroup = document.createElement('div');
        inputGroup.className = 'input-group';
        
        const label = document.createElement('label');
        label.textContent = feature;
        
        const input = document.createElement('input');
        input.type = 'number';
        input.step = 'any';
        input.id = `shap-feature-${feature}`;
        input.placeholder = `输入 ${feature} 的值`;
        
        inputGroup.appendChild(label);
        inputGroup.appendChild(input);
        container.appendChild(inputGroup);
    });
}

// 创建SHAP与统计量分析特征输入表单
function createShapStatsFeatureInputs() {
    const container = document.getElementById('shap-stats-feature-inputs');
    container.innerHTML = '';
    
    featureNames.forEach(feature => {
        const inputGroup = document.createElement('div');
        inputGroup.className = 'input-group';
        
        const label = document.createElement('label');
        label.textContent = feature;
        
        const input = document.createElement('input');
        input.type = 'number';
        input.step = 'any';
        input.id = `shap-stats-feature-${feature}`;
        input.placeholder = `输入 ${feature} 的值`;
        
        inputGroup.appendChild(label);
        inputGroup.appendChild(input);
        container.appendChild(inputGroup);
    });
}

// 获取随机样本
async function getRandomSample() {
    const count = document.getElementById('sample-count').value;
    
    try {
        showLoading(true);
        const response = await fetch(`${API_BASE}/sample?count=${count}`);
        const data = await response.json();
        
        displayRandomSample(data.samples);
    } catch (error) {
        showError('获取随机样本失败: ' + error.message);
    } finally {
        showLoading(false);
    }
}

// 显示随机样本
function displayRandomSample(samples) {
    const container = document.getElementById('random-sample-data');
    container.innerHTML = '<h3>随机样本数据:</h3>';
    
    samples.forEach((sample, index) => {
        const sampleDiv = document.createElement('div');
        sampleDiv.className = 'result-container';
        
        let html = `<h4>样本 ${index + 1}</h4>`;
        html += '<table style="width: 100%; border-collapse: collapse;">';
        
        // 只显示前10个特征，避免表格过长
        const featuresToShow = Object.keys(sample).slice(0, 10);
        featuresToShow.forEach(feature => {
            if (feature !== 'Bankrupt') {  // 排除目标变量
                html += `<tr><td style="padding: 5px; border: 1px solid #ddd;">${feature}</td><td style="padding: 5px; border: 1px solid #ddd;">${sample[feature]}</td></tr>`;
            }
        });
        
        if (Object.keys(sample).length > 11) {  // 如果有超过10个特征
            html += `<tr><td colspan="2" style="padding: 5px; border: 1px solid #ddd; text-align: center;">... 还有 ${Object.keys(sample).length - 11} 个特征</td></tr>`;
        }
        
        html += '</table>';
        sampleDiv.innerHTML = html;
        container.appendChild(sampleDiv);
    });
}

// 预测随机样本
async function predictRandomSample() {
    const count = document.getElementById('sample-count').value;
    
    try {
        showLoading(true);
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                sample_count: parseInt(count)
            })
        });
        
        const data = await response.json();
        displayPredictionResults(data);
        
        // 获取SHAP分析
        await getShapAnalysis(count);
    } catch (error) {
        showError('预测失败: ' + error.message);
    } finally {
        showLoading(false);
    }
}

// 预测手动输入
async function predictManualInput() {
    const features = {};
    
    featureNames.forEach(feature => {
        const input = document.getElementById(`feature-${feature}`);
        const value = parseFloat(input.value);
        
        if (isNaN(value)) {
            throw new Error(`请输入有效的 ${feature} 值`);
        }
        
        features[feature] = value;
    });
    
    try {
        showLoading(true);
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                features: [features]
            })
        });
        
        const data = await response.json();
        displayPredictionResults(data);
        
        // 获取SHAP分析
        await getShapAnalysis(1, [features]);
    } catch (error) {
        showError('预测失败: ' + error.message);
    } finally {
        showLoading(false);
    }
}

// 显示预测结果
function displayPredictionResults(data) {
    const container = document.getElementById('prediction-results');
    container.innerHTML = '<h2>预测结果</h2>';
    
    data.predictions.forEach((prediction, index) => {
        const resultDiv = document.createElement('div');
        resultDiv.className = 'result-container';
        
        const probability = data.probabilities[index];
        const predictionText = prediction === 1 ? '破产' : '正常';
        const predictionColor = prediction === 1 ? '#e74c3c' : '#2ecc71';
        
        let html = `
            <div class="prediction-result">
                <h3>样本 ${index + 1}</h3>
                <p><strong>预测结果:</strong> <span style="color: ${predictionColor}">${predictionText}</span></p>
                <p><strong>破产概率:</strong> ${(probability * 100).toFixed(2)}%</p>
            </div>
        `;
        
        resultDiv.innerHTML = html;
        container.appendChild(resultDiv);
    });
}

// 获取SHAP分析
async function getShapAnalysis(sampleCount, features = null) {
    try {
        showLoading(true);
        
        let requestBody;
        if (features) {
            requestBody = { features: features };
        } else {
            requestBody = { sample_count: parseInt(sampleCount) };
        }
        
        const response = await fetch(`${API_BASE}/shap-analysis`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });
        
        const data = await response.json();
        displayShapResults(data);
    } catch (error) {
        showError('SHAP分析失败: ' + error.message);
    } finally {
        showLoading(false);
    }
}

// 显示SHAP分析结果
function displayShapResults(data) {
    const container = document.getElementById('prediction-results');
    
    // 如果容器中没有预测结果，先添加一个标题
    if (!container.innerHTML.includes('预测结果')) {
        container.innerHTML = '<h2>SHAP分析结果</h2>';
    }
    
    data.shap_values.forEach((shapValues, index) => {
        const resultDiv = document.createElement('div');
        resultDiv.className = 'result-container';
        
        let html = `
            <div class="prediction-result">
                <h3>样本 ${index + 1} 的SHAP分析</h3>
                <p><strong>基准值:</strong> ${data.base_value.toFixed(4)}</p>
            </div>
        `;
        
        // 添加SHAP解释
        html += '<div class="feature-importance">';
        html += '<h4>特征重要性 (SHAP值)</h4>';
        
        // 计算每个特征的SHAP值
        const featureImportance = [];
        
        data.feature_names.forEach((name, i) => {
            featureImportance.push({
                name: name,
                value: shapValues[i]
            });
        });
        
        // 按SHAP值绝对值排序
        featureImportance.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
        
        // 显示前10个最重要的特征
        featureImportance.slice(0, 10).forEach(feature => {
            const isPositive = feature.value >= 0;
            const barWidth = Math.min(Math.abs(feature.value) * 100, 300); // 限制最大宽度
            
            html += `
                <div class="feature-bar">
                    <span class="feature-name">${feature.name}:</span>
                    <span class="shap-bar ${isPositive ? 'positive' : 'negative'}" style="width: ${barWidth}px;"></span>
                    <span class="shap-value">${feature.value.toFixed(4)}</span>
                </div>
            `;
        });
        
        html += '</div>';
        resultDiv.innerHTML = html;
        container.appendChild(resultDiv);
    });
}

// 切换标签页
function openTab(evt, tabName) {
    const tabcontent = document.getElementsByClassName("tabcontent");
    for (let i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
    }
    
    const tablinks = document.getElementsByClassName("tablinks");
    for (let i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
    }
    
    document.getElementById(tabName).style.display = "block";
    evt.currentTarget.className += " active";
}

// 显示/隐藏加载状态
function showLoading(show) {
    document.getElementById('loading').style.display = show ? 'block' : 'none';
}

// 显示错误信息
function showError(message) {
    const errorElement = document.getElementById('error-message');
    errorElement.textContent = message;
    errorElement.style.display = 'block';
    
    // 5秒后自动隐藏错误信息
    setTimeout(() => {
        errorElement.style.display = 'none';
    }, 5000);
}

// 获取SHAP分析的随机样本
async function getRandomSampleForShap() {
    const count = document.getElementById('shap-sample-count').value;
    
    try {
        showLoading(true);
        const response = await fetch(`${API_BASE}/sample?count=${count}`);
        const data = await response.json();
        
        displayShapRandomSample(data.samples);
    } catch (error) {
        showError('获取随机样本失败: ' + error.message);
    } finally {
        showLoading(false);
    }
}

// 显示SHAP分析的随机样本
function displayShapRandomSample(samples) {
    const container = document.getElementById('shap-random-sample-data');
    container.innerHTML = '<h3>随机样本数据:</h3>';
    
    samples.forEach((sample, index) => {
        const sampleDiv = document.createElement('div');
        sampleDiv.className = 'result-container';
        
        let html = `<h4>样本 ${index + 1}</h4>`;
        html += '<table style="width: 100%; border-collapse: collapse;">';
        
        // 只显示前10个特征，避免表格过长
        const featuresToShow = Object.keys(sample).slice(0, 10);
        featuresToShow.forEach(feature => {
            if (feature !== 'Bankrupt') {  // 排除目标变量
                html += `<tr><td style="padding: 5px; border: 1px solid #ddd;">${feature}</td><td style="padding: 5px; border: 1px solid #ddd;">${sample[feature]}</td></tr>`;
            }
        });
        
        if (Object.keys(sample).length > 11) {  // 如果有超过10个特征
            html += `<tr><td colspan="2" style="padding: 5px; border: 1px solid #ddd; text-align: center;">... 还有 ${Object.keys(sample).length - 11} 个特征</td></tr>`;
        }
        
        html += '</table>';
        sampleDiv.innerHTML = html;
        container.appendChild(sampleDiv);
    });
}

// 分析随机样本
async function analyzeRandomSample() {
    const count = document.getElementById('shap-sample-count').value;
    
    try {
        showLoading(true);
        await getShapAnalysis(count);
    } catch (error) {
        showError('SHAP分析失败: ' + error.message);
    } finally {
        showLoading(false);
    }
}

// 分析手动输入
async function analyzeManualInput() {
    const features = {};
    
    featureNames.forEach(feature => {
        const input = document.getElementById(`shap-feature-${feature}`);
        const value = parseFloat(input.value);
        
        if (isNaN(value)) {
            throw new Error(`请输入有效的 ${feature} 值`);
        }
        
        features[feature] = value;
    });
    
    try {
        showLoading(true);
        await getShapAnalysis(1, [features]);
    } catch (error) {
        showError('SHAP分析失败: ' + error.message);
    } finally {
        showLoading(false);
    }
}

// 获取SHAP与统计量分析的随机样本
async function getRandomSampleForShapStats() {
    const count = document.getElementById('shap-stats-sample-count').value;
    
    try {
        showLoading(true);
        const response = await fetch(`${API_BASE}/sample?count=${count}`);
        const data = await response.json();
        
        displayShapStatsRandomSample(data.samples);
    } catch (error) {
        showError('获取随机样本失败: ' + error.message);
    } finally {
        showLoading(false);
    }
}

// 显示SHAP与统计量分析的随机样本
function displayShapStatsRandomSample(samples) {
    const container = document.getElementById('shap-stats-random-sample-data');
    container.innerHTML = '<h3>随机样本数据:</h3>';
    
    samples.forEach((sample, index) => {
        const sampleDiv = document.createElement('div');
        sampleDiv.className = 'result-container';
        
        let html = `<h4>样本 ${index + 1}</h4>`;
        html += '<table style="width: 100%; border-collapse: collapse;">';
        
        // 只显示前10个特征，避免表格过长
        const featuresToShow = Object.keys(sample).slice(0, 10);
        featuresToShow.forEach(feature => {
            if (feature !== 'Bankrupt') {  // 排除目标变量
                html += `<tr><td style="padding: 5px; border: 1px solid #ddd;">${feature}</td><td style="padding: 5px; border: 1px solid #ddd;">${sample[feature]}</td></tr>`;
            }
        });
        
        if (Object.keys(sample).length > 11) {  // 如果有超过10个特征
            html += `<tr><td colspan="2" style="padding: 5px; border: 1px solid #ddd; text-align: center;">... 还有 ${Object.keys(sample).length - 11} 个特征</td></tr>`;
        }
        
        html += '</table>';
        sampleDiv.innerHTML = html;
        container.appendChild(sampleDiv);
    });
}

// 分析随机样本（带统计量）
async function analyzeRandomSampleWithStats() {
    const count = document.getElementById('shap-stats-sample-count').value;
    
    try {
        showLoading(true);
        await getShapWithStatsAnalysis(count);
    } catch (error) {
        showError('SHAP与统计量分析失败: ' + error.message);
    } finally {
        showLoading(false);
    }
}

// 分析手动输入（带统计量）
async function analyzeManualInputWithStats() {
    const features = {};
    
    featureNames.forEach(feature => {
        const input = document.getElementById(`shap-stats-feature-${feature}`);
        const value = parseFloat(input.value);
        
        if (isNaN(value)) {
            throw new Error(`请输入有效的 ${feature} 值`);
        }
        
        features[feature] = value;
    });
    
    try {
        showLoading(true);
        await getShapWithStatsAnalysis(1, [features]);
    } catch (error) {
        showError('SHAP与统计量分析失败: ' + error.message);
    } finally {
        showLoading(false);
    }
}

// 获取SHAP与统计量分析
async function getShapWithStatsAnalysis(sampleCount, features = null) {
    try {
        showLoading(true);
        
        let requestBody;
        if (features) {
            requestBody = { features: features };
        } else {
            requestBody = { sample_count: parseInt(sampleCount) };
        }
        
        // 使用WebSocket进行流式分析
        await performStreamingShapAnalysis(requestBody);
    } catch (error) {
        showError('SHAP与统计量分析失败: ' + error.message);
    } finally {
        showLoading(false);
    }
}

// 使用WebSocket进行流式SHAP分析
async function performStreamingShapAnalysis(requestBody) {
    return new Promise((resolve, reject) => {
        // 构建WebSocket URL
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsHost = window.location.hostname;
        const wsPort = window.location.port || (window.location.protocol === 'https:' ? '443' : '80');
        const wsUrl = `${wsProtocol}//${wsHost}:${wsPort}/ws/shap-with-stats-analysis`;
        
        const ws = new WebSocket(wsUrl);
        let shapData = null;
        let llmInterpretationDiv = null;
        
        ws.onopen = function() {
            console.log('WebSocket连接已建立');
            // 发送请求数据
            ws.send(JSON.stringify(requestBody));
        };
        
        ws.onmessage = function(event) {
            try {
                const message = JSON.parse(event.data);
                console.log('收到WebSocket消息:', message.type, message);
                
                if (message.type === 'status') {
                    // 更新状态显示
                    console.log('状态更新:', message.message, '时间:', new Date().toLocaleTimeString());
                    const loadingElement = document.querySelector('.loading');
                    if (loadingElement) {
                        loadingElement.textContent = message.message;
                    }
                    
                } else if (message.type === 'shap_results') {
                    // 接收到SHAP数据，异步显示基础结果
                    console.log('接收到SHAP结果:', '时间:', new Date().toLocaleTimeString());
                    shapData = message.data;
                    
                    // 更新状态显示
                    const loadingElement = document.querySelector('.loading');
                    if (loadingElement) {
                        loadingElement.textContent = '正在渲染SHAP结果...';
                    }
                    
                    // 异步处理SHAP结果显示，避免阻塞后续消息
                    setTimeout(() => {
                        displayShapResults(shapData);
                        
                        // 准备LLM解读容器
                        const container = document.getElementById('prediction-results');
                        const llmDiv = document.createElement('div');
                        llmDiv.className = 'result-container';
                        llmDiv.innerHTML = `
                            <div class="prediction-result">
                                <h3>LLM解读</h3>
                                <div id="llm-interpretation" style="white-space: pre-line; font-family: monospace; border: 1px solid #ddd; padding: 10px; background-color: #f9f9f9; min-height: 100px;"></div>
                            </div>
                        `;
                        container.appendChild(llmDiv);
                        llmInterpretationDiv = document.getElementById('llm-interpretation');
                    }, 10); // 很短的延迟，让其他消息有机会被处理
                    
                } else if (message.type === 'llm_start') {
                    // LLM开始解读
                    console.log('LLM开始解读:', message.message, '时间:', new Date().toLocaleTimeString());
                    // 更新加载状态显示
                    const loadingElement = document.querySelector('.loading');
                    if (loadingElement) {
                        loadingElement.textContent = message.message;
                    }
                    
                } else if (message.type === 'llm_chunk') {
                    // 接收到LLM流式数据块
                    if (llmInterpretationDiv && message.content) {
                        // 创建文本节点并追加，实现真正的流式显示
                        const textNode = document.createTextNode(message.content);
                        llmInterpretationDiv.appendChild(textNode);
                        // 自动滚动到底部
                        llmInterpretationDiv.scrollTop = llmInterpretationDiv.scrollHeight;
                        
                        // 在控制台同步显示，与后端输出保持一致
                        console.log(message.content);
                    }
                    
                } else if (message.type === 'llm_complete') {
                    // LLM解读完成
                    console.log('LLM解读完成');
                    ws.close();
                    resolve();
                    
                } else if (message.type === 'error') {
                    // 发生错误
                    console.error('WebSocket错误:', message.message);
                    reject(new Error(message.message));
                    ws.close();
                }
            } catch (error) {
                console.error('解析WebSocket消息失败:', error);
                reject(error);
            }
        };
        
        ws.onerror = function(error) {
            console.error('WebSocket连接错误:', error);
            reject(new Error('WebSocket连接失败'));
        };
        
        ws.onclose = function() {
            console.log('WebSocket连接已关闭');
        };
    });
}